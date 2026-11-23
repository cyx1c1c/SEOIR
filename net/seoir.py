import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath

BIAS_FREE = 'BiasFree'
WITH_BIAS = 'WithBias'

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == BIAS_FREE:
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, 
            hidden_features * 2, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=hidden_features * 2,
            bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GatedCNNBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        expansion_ratio=8/3, 
        kernel_size=7, 
        conv_ratio=1.0,
        norm_layer=nn.BatchNorm2d, 
        act_layer=nn.GELU,
        drop_path=0.,
        **kwargs
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.gic1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.cconv = nn.Conv2d(
            conv_channels, 
            conv_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=conv_channels
        )
        self.gic2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        gic = self.gic1(x)
        g, i, c = torch.split(gic, self.split_indices, dim=1)
        c = self.cconv(c)
        x = self.gic2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        x = x + shortcut
        x = self.proj_out(x)
        return x

class GatedDegradationAdaption(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 8/3, 
        kernel_size: int = 7,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
        bias: bool = False,
        eps: float = 1e-6,
        task_num: int = 5,
    ):
        super().__init__()
        assert expansion_ratio > 2.0, "expansion_ratio > 2 to ensure γ、β、α"
        self.eps = eps
        self.norm = norm_layer(dim)
        self.act = act_layer()
        hidden = int(round(expansion_ratio * dim))
        gamma_ch = dim
        beta_ch  = dim
        alpha_ch = hidden - gamma_ch - beta_ch
        assert alpha_ch > 0, "alpha channel number > 0, please enlarge expansion_ratio"
        self.W_exp = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.W_depth = nn.Conv2d(
            alpha_ch, alpha_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=alpha_ch,
            bias=bias
        )
        self.W_gate = nn.Conv2d(beta_ch + alpha_ch, dim, kernel_size=1, bias=bias)
        self.W_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.tau = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.dw_proj = nn.Linear(task_num, dim)
        self.split_indices = (gamma_ch, beta_ch, alpha_ch)

    def forward(self, x, dw=None):
        F_in = x
        F_hat = self.norm(F_in)
        mu = F_hat.mean(dim=(2, 3), keepdim=True)
        var = ((F_hat - mu) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma = torch.sqrt(var + self.eps)
        Delta = torch.sigmoid(mu + sigma)
        tau_adj = self.tau * Delta
        if dw is not None:
            task_scale = torch.sigmoid(self.dw_proj(dw)).view(x.shape[0], -1, 1, 1)
            tau_adj = tau_adj * task_scale
        F_prime = self.W_exp(F_hat)
        gamma, beta, alpha = torch.split(F_prime, self.split_indices, dim=1)
        alpha_dw = self.W_depth(alpha)
        if alpha_dw.shape[1] != tau_adj.shape[1]:
            repeat_factor = (alpha_dw.shape[1] + tau_adj.shape[1] - 1) // tau_adj.shape[1]
            tau_alpha = tau_adj.repeat(1, repeat_factor, 1, 1)[:, :alpha_dw.shape[1]]
        else:
            tau_alpha = tau_adj
        alpha_prime = alpha_dw * (1.0 + tau_alpha)
        gate = torch.sigmoid(gamma)
        merged = torch.cat([beta, alpha_prime], dim=1)
        F_gate = gate * self.W_gate(merged)
        F_out = self.W_proj(F_gate + F_in)
        return F_out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        assert dim % 2 == 0, "dim must be even (split att/gate)"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        half_dim = dim // 2
        self.qkv = nn.Conv2d(half_dim, half_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            half_dim * 3, half_dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=half_dim * 3, bias=bias
        )
        self.gatedcnn = GatedDegradationAdaption(dim=half_dim, expansion_ratio=8/3, kernel_size=7)
        self.lam_net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.lam_net[-2].weight)
        nn.init.zeros_(self.lam_net[-2].bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    @staticmethod
    def _spatial_fusion(F_att, F_gate):
        F_ag = F_att + torch.sigmoid(F_gate)
        F_ga = F_gate + torch.sigmoid(F_att)
        F_s = torch.cat([F_ag, F_ga], dim=1)
        return F_s

    @staticmethod
    def _frequency_fusion(F_att, F_gate, size_hw):
        Fhat_att = torch.fft.rfft2(F_att)
        Fhat_gate = torch.fft.rfft2(F_gate)
        Fhat = Fhat_att + Fhat_gate
        F_freq = torch.fft.irfft2(Fhat, s=size_hw)
        F_freq_rep = F_freq.repeat_interleave(2, dim=1)
        return F_freq_rep


    def forward(self, x, dw=None):
        x_att = x[:, 0::2, :, :]
        x_gate = x[:, 1::2, :, :]
        b, _, h, w = x_att.shape
        qkv = self.qkv_dwconv(self.qkv(x_att))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out_att = attn @ v
        out_att = rearrange(out_att, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_gate = self.gatedcnn(x_gate, dw)
        Fs = self._spatial_fusion(out_att, out_gate)
        Ff = self._frequency_fusion(out_att, out_gate, (h, w))
        lam_map = self.lam_net(torch.cat([out_att, out_gate], dim=1))
        F_fuse = lam_map * Fs + (1.0 - lam_map) * Ff
        out = self.project_out(F_fuse)
        return out


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, dw=None):
        x = x + self.attn(self.norm1(x), dw)
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class SEOIR(nn.Module):
    """
    SEOIR: A neural network for image restoration.
    
    This model uses a UNet-like architecture with transformer blocks for feature extraction
    and processing at multiple resolutions.
    
    Args:
        inp_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 3)
        dim (int): Base dimension for feature maps (default: 24)
        num_blocks (list): Number of transformer blocks at each resolution (default: [3,5,5,7])
        num_refinement_blocks (int): Number of refinement blocks (default: 2)
        heads (list): Number of attention heads at each resolution (default: [1,2,4,8])
        ffn_expansion_factor (float): Expansion factor for FFN (default: 2)
        bias (bool): Whether to use bias in convolutions (default: False)
        LayerNorm_type (str): Type of layer normalization (default: 'WithBias')
    """
    
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=28,
        num_blocks=[3, 5, 5, 7],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type=WITH_BIAS,
    ):
        super().__init__()

        d1, d2, d3, d4 = dim, dim * 2, dim * 4, dim * 8

        def make_stage(stage_dim, stage_heads, stage_blocks):
            return nn.ModuleList([
                TransformerBlock(
                    dim=stage_dim,
                    num_heads=stage_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                )
                for _ in range(stage_blocks)
            ])

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = make_stage(d1, heads[0], num_blocks[0])

        self.down1_2 = Downsample(d1)
        self.encoder_level2 = make_stage(d2, heads[1], num_blocks[1])

        self.down2_3 = Downsample(d2)
        self.encoder_level3 = make_stage(d3, heads[2], num_blocks[2])


        self.down3_4 = Downsample(d3)
        self.latent = make_stage(d4, heads[3], num_blocks[3])


        self.up4_3 = Upsample(d3)
        self.reduce_chan_level3 = nn.Conv2d(d2 + d3, d3, kernel_size=1, bias=bias)
        self.reduce_dim_level3 = nn.Conv2d(d4, d3, kernel_size=1, bias=bias)
        self.decoder_level3 = make_stage(d3, heads[2], num_blocks[2])

        self.up3_2 = Upsample(d3)
        self.reduce_chan_level2 = nn.Conv2d(d3, d2, kernel_size=1, bias=bias)
        self.decoder_level2 = make_stage(d2, heads[1], num_blocks[1])

        self.up2_1 = Upsample(d2)
        self.decoder_level1 = make_stage(d2, heads[0], num_blocks[0])

        self.refinement = make_stage(d2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(d2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.deg_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d4, 5),
            nn.Softmax(dim=-1),
        )

    def forward(self, inp_img, noise_emb=None):
        """
        Forward pass of the SEOIR network.
        
        Args:
            inp_img (Tensor): Input image tensor [B, inp_channels, H, W]
            noise_emb (Tensor, optional): Noise embedding tensor (not used in this implementation)
            
        Returns:
            Tensor: Restored image tensor [B, out_channels, H, W]
        """

        # --- Initial feature extraction
        inp_enc_level1 = self.patch_embed(inp_img)

        def run_stage(x, stage, dw=None):
            for blk in stage:
                x = blk(x, dw)
            return x

        # --- Encoder path (downsampling)
        out_enc_level1 = run_stage(inp_enc_level1, self.encoder_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = run_stage(inp_enc_level2, self.encoder_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = run_stage(inp_enc_level3, self.encoder_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        dw = self.deg_classifier(inp_enc_level4)
        latent = run_stage(inp_enc_level4, self.latent, dw=dw)

        # --- Prepare bottleneck features for decoder
        latent = self.reduce_dim_level3(latent)

        # --- Decoder path (upsampling with skip connections)
        # Level 4 to 3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = run_stage(inp_dec_level3, self.decoder_level3, dw=dw)

        # Level 3 to 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = run_stage(inp_dec_level2, self.decoder_level2, dw=dw)

        # Level 2 to 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = run_stage(inp_dec_level1, self.decoder_level1, dw=dw)

        # --- Final refinement
        out_dec_level1 = run_stage(out_dec_level1, self.refinement, dw=dw)

        # Output with residual connection to input
        output = self.output(out_dec_level1) + inp_img

        return output


# Calculating Parameters, FLOPs, Memory
if __name__ == "__main__":
    # Create model instance
    model = SEOIR(
        num_blocks=[3, 5, 5, 7],
        dim=28,
        ffn_expansion_factor=2,
        num_refinement_blocks=4
    ).cuda()

    # Create test input
    x = torch.randn(1, 3, 224, 224).cuda()

    # Print model architecture
    print(model)
    model.eval()
    with torch.no_grad():
        out = model(x)

    # Print memory usage
    print('{:>16s} : {:<.3f} [M]'.format(
        'Max Memory', 
        torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2
    ))
    print(f'Output shape: {out.shape}')

    # Calculate and print FLOPS and parameters
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))
