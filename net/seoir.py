"""
SEOIR: Simplex-Evolutionary Optimization for All-in-One Image Restoration

本文件实现了 SEOIR 网络模型的核心架构，包括：
- 层归一化（带/不带偏置）
- 前馈网络（FeedForward）
- 门控卷积块（GatedCNNBlock）
- 门控退化自适应模块（GatedDegradationAdaption）
- 多头自注意力（Attention，含空间-频率融合）
- Transformer 块（TransformerBlock）
- 上下采样模块（Downsample/Upsample）
- 主干网络 SEOIR（U-Net 结构）

20260901 相对 20260726 的改动：
  将全局标量 λ 升级为空间自适应 λ(x)：
    - lam_net: 基于两个分支输出 (out_att + out_gate) 生成逐像素融合权重图 (B,1,H,W)
    - 零初始化最后一层 Conv2d → sigmoid(0)=0.5，训练初始等价于基线 λ=0.5
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # 用于灵活的张量维度重排
from fvcore.nn import FlopCountAnalysis, flop_count_table  # 计算 FLOPs 和参数量
from timm.models.layers import DropPath   # 随机深度（Stochastic Depth）


# ==================== 常量定义 ====================
# Constants
BIAS_FREE = 'BiasFree'  # 无偏置的层归一化类型
WITH_BIAS = 'WithBias'  # 带偏置的层归一化类型


# ==================== 实用工具函数 ====================
# Utility Functions
def to_3d(x):
    """Reshape tensor from 4D to 3D."""
    """将 4D 张量 (b, c, h, w) 转换为 3D 张量 (b, h*w, c)"""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Reshape tensor from 3D back to 4D."""
    """将 3D 张量 (b, h*w, c) 恢复为 4D 张量 (b, c, h, w)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ==================== 层归一化模块 ====================
# Layer Normalization Components
class BiasFree_LayerNorm(nn.Module):
    """不带偏置的层归一化，只学习缩放因子 weight。
       输入形状: (b, c, h, w) 或 (b, n, c) 等，归一化在最后一个维度上进行。
    """
    """Layer normalization without bias."""
    
    def __init__(self, normalized_shape):
        super().__init__()
        # 如果传入的是整数，则转换为单元素元组
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        # 确保归一化的形状是一维的（即只对最后一个维度进行归一化）
        assert len(normalized_shape) == 1
        
        # 可学习的缩放参数（weight），初始为全1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        # 计算方差（unbiased=False 表示使用有偏估计，即除以 N）
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 标准化： (x - μ) / σ，但这里没有减去均值（因为假设输入已经中心化？实际上常见 LayerNorm 会减均值）
        # 注意：此实现直接除以标准差，没有减去均值，意味着它假设输入均值为0。
        # 在某些情况下这可能成立，但标准的 LayerNorm 通常包含减均值操作。
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化，学习缩放因子 weight 和偏移 bias。"""
    """Layer normalization with bias."""
    
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
        # 计算均值
        mu = x.mean(-1, keepdim=True)
        # 计算方差（有偏估计）
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 标准化： (x - μ) / σ * weight + bias
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """层归一化工厂类，根据传入的类型选择无偏置或有偏置的归一化。"""
    """Layer normalization factory that creates either biased or unbiased norm."""
    
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == BIAS_FREE:
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    
    def forward(self, x):
        # 输入 x 是 4D 张量 (b, c, h, w)
        h, w = x.shape[-2:]
        # 转换为 3D (b, h*w, c) -> 归一化 -> 再转回 4D
        return to_4d(self.body(to_3d(x)), h, w)


# ==================== 神经网络基础构件 ====================
# Neural Network Building Blocks
class FeedForward(nn.Module):
    """Feed Forward Network with GELU activation and depthwise convolution."""
    """前馈网络（FFN）：1x1 卷积扩展通道 -> 深度可分离卷积（3x3）-> GELU 门控 -> 1x1 卷积投影回原通道。
       与 Transformer 中常见的 FFN 类似，但加入了门控机制和深度卷积。
    """
    
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)   # 扩展后的通道数
        
        # 第一层：1x1 卷积将通道数扩展到 2 * hidden_features
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积（3x3，分组数等于输入通道数），保持通道数不变
        self.dwconv = nn.Conv2d(
            hidden_features * 2, 
            hidden_features * 2, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=hidden_features * 2,   # 分组卷积，每个通道单独卷积
            bias=bias
        )
        # 最后一层：1x1 卷积将通道数映射回原维度 dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        # x: (b, dim, h, w)
        x = self.project_in(x)  # (b, 2*hidden, h, w)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 分成两个 (b, hidden, h, w)
        # 门控：x1 经过 GELU 激活后与 x2 逐元素相乘
        x = F.gelu(x1) * x2
        x = self.project_out(x)  # (b, dim, h, w)
        return x


class GatedCNNBlock(nn.Module):
    """Gated CNN Block implementation based on https://arxiv.org/pdf/1612.08083"""
    """门控 CNN 块，基于 https://arxiv.org/pdf/1612.08083 的思想。
       包含：归一化 -> 1x1 卷积扩展 -> 拆分为 g, i, c -> c 部分经过深度卷积 -> 门控组合 -> 残差 + 输出投影。
       本模块在代码中定义但未在最终 SEOIR 中使用（可能为历史遗留），实际使用的是 GatedDegradationAdaption。
    """
    
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
        self.norm = norm_layer(dim)  # 归一化层，默认 BatchNorm2d
        hidden = int(expansion_ratio * dim)  # 扩展后的通道数
        # 1x1 卷积将通道扩展到 2*hidden（因为后面要拆分为 g, i, c）
        self.gic1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.act = act_layer()
        
        # 用于决定 c 的通道数，即 conv_ratio * dim
        conv_channels = int(conv_ratio * dim)
        # 定义拆分尺寸：(g 通道数, i 通道数, c 通道数)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # 对 c 部分进行深度卷积（分组数等于 conv_channels）
        self.cconv = nn.Conv2d(
            conv_channels, 
            conv_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=conv_channels
        )
        # 1x1 卷积将 hidden 通道映射回 dim
        self.gic2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        # 随机深度（Stochastic Depth），如果 drop_path > 0 则使用
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 最后的输出投影（1x1 卷积），保持维度 dim
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x):
        shortcut = x
        x = self.norm(x)  # (b, dim, h, w)

        gic = self.gic1(x)  # [B, 2 * hidden, H, W]
        # 拆分为 g, i, c，各自形状为 (b, hidden), (b, hidden-conv_channels), (b, conv_channels)
        g, i, c = torch.split(gic, self.split_indices, dim=1)  # [B, x, H, W] for each
        c = self.cconv(c)  # [B, conv_channels, H, W]   对 c 做深度卷积
        # 门控：g 经过激活，与 i 和 c 拼接后相乘，然后通过 gic2 映射回 dim
        x = self.gic2(self.act(g) * torch.cat((i, c), dim=1))  # [B, C, H, W]
        x = self.drop_path(x)

        x = x + shortcut
        x = self.proj_out(x)  # 最终输出投影

        return x

class GatedDegradationAdaption(nn.Module):
    """
    Algorithm 1: Gated Degradation Adaption

    Input:  F_in^gate  ∈ R^{C×H×W}
    Output:  F_out^gate ∈ R^{C×H×W}

    """
    
    """
    算法 1：门控退化自适应模块（Gated Degradation Adaptation）
    输入:  F_in^gate  ∈ R^{C×H×W}
    输出: F_out^gate ∈ R^{C×H×W}

    核心思想：
    - 使用可学习参数 τ 和空间统计信息（均值、方差）动态调整卷积的贡献。
    - 将特征分为 γ, β, α 三部分，其中 γ 用于生成门控信号，β 保持不变，α 经过深度卷积后由 τ 调制。
    - 最终通过残差连接和投影输出。
    """
    
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
        # 确保扩展因子足够大，使得 α 的通道数 > 0
        assert expansion_ratio > 2.0, "expansion_ratio > 2 to ensure γ、β、α"

        self.eps = eps
        self.norm = norm_layer(dim)  # 归一化层，默认 BatchNorm2d
        self.act = act_layer()  # 激活函数，默认 GELU

        # ---- W_exp：expand and then split γ、β、α
        # ---- W_exp：扩展并拆分为 γ, β, α
        hidden = int(round(expansion_ratio * dim))
        gamma_ch = dim  # γ 通道数 = C
        beta_ch  = dim  # β 通道数 = C
        alpha_ch = hidden - gamma_ch - beta_ch  # α 通道数 = hidden - 2C
        assert alpha_ch > 0, "alpha channel number > 0, please enlarge expansion_ratio"

        # 1x1 卷积将通道从 dim 扩展到 hidden
        self.W_exp = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)

        # ---- W_depth：depthwise conv for α
        # ---- W_depth：对 α 部分进行深度卷积
        self.W_depth = nn.Conv2d(
            alpha_ch, alpha_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=alpha_ch,  # 分组数等于通道数，实现深度可分离卷积
            bias=bias
        )

        # ---- W_gate：concat(β, α') and then back to C
        # ---- W_gate：将 β 和 α' 拼接后映射回 dim
        self.W_gate = nn.Conv2d(beta_ch + alpha_ch, dim, kernel_size=1, bias=bias)

        # ---- W_proj：residual first and then 1×1 mapping
        # ---- W_proj：残差后最后的 1x1 投影
        self.W_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # init τ(learnable)
        # 可学习参数 τ（形状 1 x C x 1 x 1），用于调制 α' 的贡献
        self.tau = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.dw_proj = nn.Linear(task_num, dim) # 任务依赖的可学习参数，用于调整 α' 的贡献

        # 记录拆分尺寸，用于前向传播
        self.split_indices = (gamma_ch, beta_ch, alpha_ch)

    def forward(self, x, dw=None):
        F_in = x

        # 1) Normalize
        # 1) 归一化
        F_hat = self.norm(F_in)

        # 2) spatial mu and var per channel
        # 2) 计算每个通道的空间均值和方差
        mu = F_hat.mean(dim=(2, 3), keepdim=True)                        # [B,C,1,1]
        var = ((F_hat - mu) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma = torch.sqrt(var + self.eps)

        # 3) Δ = Sigmoid(mu + sigma)
        Delta = torch.sigmoid(mu + sigma)

        # 4) τ_adj = τ * Δ  （即用 Δ 调整 τ）
        tau_adj = self.tau * Delta
        if dw is not None:
        # dw: (B, task_num) -> (B, dim, 1, 1)，sigmoid 保证缩放在 (0,1)
            task_scale = torch.sigmoid(self.dw_proj(dw)).view(x.shape[0], -1, 1, 1)
            tau_adj = tau_adj * task_scale

        # 5) F' = F_hat W_exp
        F_prime = self.W_exp(F_hat)  # (b, hidden, h, w)

        # 6) γ, β, α = split(F')
        # 6) 将 F' 拆分为 γ, β, α
        gamma, beta, alpha = torch.split(F_prime, self.split_indices, dim=1)

        # 7) α' = (α ⊗ W_depth) * (1 + τ_adj)
        alpha_dw = self.W_depth(alpha)

        # Broadcast to alpha channels.
        # tau_adj: [B, C, 1, 1], only scales the first C channels.
        # Clamp/slice to dim, then repeat or interpolate to match alpha channels if needed.
        # 因为 τ_adj 的形状是 (b, C, 1, 1)，而 alpha_dw 的通道数是 alpha_ch（可能不等于 C）
        # 需要通过重复来对齐通道数
        if alpha_dw.shape[1] != tau_adj.shape[1]:
            repeat_factor = (alpha_dw.shape[1] + tau_adj.shape[1] - 1) // tau_adj.shape[1]
            tau_alpha = tau_adj.repeat(1, repeat_factor, 1, 1)[:, :alpha_dw.shape[1]]
        else:
            tau_alpha = tau_adj
        alpha_prime = alpha_dw * (1.0 + tau_alpha)

        # 8) F_gate = σ(γ) * concat(β, α') W_gate
        gate = torch.sigmoid(gamma)  # 门控信号，形状 (b, gamma_ch, h, w)
        merged = torch.cat([beta, alpha_prime], dim=1)  # (b, beta_ch+alpha_ch, h, w)
        F_gate = gate * self.W_gate(merged)

        # 9) F_out = (F_gate + F_in) W_proj
        F_out = self.W_proj(F_gate + F_in)

        # 10) return
        return F_out


class Attention(nn.Module):
    """Multi-head self-attention with Spatial-Frequency Fusion (per screenshots).
       - SpatialFusion: cross-enhance + channel-wise concat
       - FrequencyFusion: rFFT2 combine + iFFT2, channel repeat to match
    """
    """多头自注意力，包含空间-频率融合算法（算法2）。
       包含：
         - 通道分割（偶数通道进 attention，奇数通道进 GatedDA）
         - attention 分支（multi-head self-attention）
         - gated 分支（GatedDegradationAdaption）
         - 空间-频率融合（SpatialFusion 和 FrequencyFusion）
         - 空间自适应加权融合 + 投影输出
    """
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        # 要求 dim 是偶数，因为要分成 attention 和 gated 两个分支（各一半）
        assert dim % 2 == 0, "dim must be even (split att/gate)"
        self.num_heads = num_heads

        # 温度参数，用于缩放注意力分数，形状 (num_heads, 1, 1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        half_dim = dim // 2  # 每个分支的通道数

        # attention 分支：生成 Q, K, V，先 1x1 卷积扩展，然后深度卷积
        self.qkv = nn.Conv2d(half_dim, half_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            half_dim * 3, half_dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=half_dim * 3, bias=bias
        )

        # gated 分支：使用 GatedDegradationAdaption
        self.gatedcnn = GatedDegradationAdaption(dim=half_dim, expansion_ratio=8/3, kernel_size=7)

        # Alg.2 Step3
        # 空间自适应融合权重生成器（替代全局标量 λ）
        # 输入：out_att + out_gate 拼接 → (B, dim, H, W)
        # 输出：(B, 1, H, W)，每个像素位置独立决定融合比例
        # 最后一层零初始化 → sigmoid(0) = 0.5，训练初始等价于基线 λ=0.5
        self.lam_net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # 零初始化最后一层 Conv2d（lam_net[-2]）：权重和偏置均为 0
        # → 初始输出 sigmoid(0) = 0.5，与基线 λ=0.5 完全等价
        nn.init.zeros_(self.lam_net[-2].weight)
        nn.init.zeros_(self.lam_net[-2].bias)

        # 最终的输出投影（将融合后的特征投影回原通道数 dim）
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    # -------------------- Algorithm 2: helpers --------------------
    # -------------------- 算法2辅助函数 --------------------
    @staticmethod
    def _spatial_fusion(F_att, F_gate):
        """空间融合：
            F^α_g = F_att + Sigmoid(F_gate)
            F^α_a = F_gate + Sigmoid(F_att)
            F_s = Concat(F^α_g, F^α_a)   # 通道拼接
        """
        # F^α_g = F_att + Sigmoid(F_gate)
        F_ag = F_att + torch.sigmoid(F_gate)
        # F^α_a = F_gate + Sigmoid(F_att)
        F_ga = F_gate + torch.sigmoid(F_att)
        # channel-wise merge: Concat(F^α_g, F^α_a)
        F_s = torch.cat([F_ag, F_ga], dim=1)
        return F_s

    @staticmethod
    def _frequency_fusion(F_att, F_gate, size_hw):
        """频率融合：
            - 对 F_att 和 F_gate 分别进行实数 FFT (rfft2)
            - 相加得到频域组合
            - 逆变换 (irfft2) 回到空域
            - 将结果重复一次以匹配空间融合的通道数（因为空间融合通道翻倍）
        """
        # real FFT
        # 实数 FFT：输入为实数，输出为复数的频域表示（只保留一半的频域分量）
        Fhat_att = torch.fft.rfft2(F_att)
        Fhat_gate = torch.fft.rfft2(F_gate)
        Fhat = Fhat_att + Fhat_gate
        # inverse FFT
        # 逆变换，需要指定输出尺寸（因为 rfft2 会损失一半信息，irfft2 需要知道原尺寸）
        F_freq = torch.fft.irfft2(Fhat, s=size_hw)
        # Repeat to match channels of SpatialFusion (which is 2 * half_dim)
        # 将通道数重复为原来的两倍（因为空间融合后通道数翻倍，这里需要对齐）
        F_freq_rep = F_freq.repeat_interleave(2, dim=1)
        return F_freq_rep
    # -------------------------------------------------------------

    def forward(self, x, dw=None):
        # 1) skip-split channels
        # 1) 跳跃拆分（skip-split）：将输入通道按奇偶分成两组
        # 偶数索引通道（0,2,4,...）进入 attention 分支
        x_att = x[:, 0::2, :, :]  # even channels   shape: (b, C/2, h, w)
        # 奇数索引通道（1,3,5,...）进入 gated 分支
        x_gate = x[:, 1::2, :, :]  # odd channels   shape: (b, C/2, h, w)

        # --- Attention path ---
        # --- Attention 路径 ---
        b, _, h, w = x_att.shape
        # 1x1 卷积 + 深度卷积得到 Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x_att))
        q, k, v = qkv.chunk(3, dim=1)  # 每个 (b, C/2, h, w)
        # 将空间维度展开，并重新排列为多头形式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 对 Q 和 K 进行 L2 归一化（提升稳定性）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # 计算注意力分数：Q·K^T * temperature，然后 softmax
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 加权求和得到输出
        out_att = attn @ v
        # 恢复形状： (b, head*c, h, w)
        out_att = rearrange(out_att, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # --- Gated (degradation) path ---
        # --- Gated 路径（退化自适应）---
        out_gate = self.gatedcnn(x_gate, dw)

        # --- Algorithm 2: Spatial-Frequency Fusion ---
        # --- 算法2：空间-频率融合 ---
        # 空间融合：输出通道数翻倍（2 * (C/2) = C）
        Fs = self._spatial_fusion(out_att, out_gate)                 # spatial cross-enhance + concat   (b, C, h, w)
        
        # 频率融合：输入通道数各为 C/2，输出通道数也为 C（重复后）
        Ff = self._frequency_fusion(out_att, out_gate, (h, w))       # rFFT combine + iFFT + repeat     (b, C, h, w)

        # 空间自适应融合权重：基于 out_att + out_gate 生成逐像素 λ(x) ∈ [0,1]
        lam_map = self.lam_net(torch.cat([out_att, out_gate], dim=1))  # (b, 1, h, w)，自动广播

        # 加权融合（每个像素独立决定信空间融合还是频域融合）
        F_fuse = lam_map * Fs + (1.0 - lam_map) * Ff                  # spatial adaptive fusion          (b, C, h, w)
        
        # 最终投影输出（可选的残差连接在外层实现）
        out = self.project_out(F_fuse)                               # residual/project in outer block if needed   (b, C, h, w)
        return out


class ResBlock(nn.Module):
    """Residual block with two convolutions."""
    """简单的残差块：两个 3x3 卷积，中间 PReLU，残差连接。
       注意：此模块在代码中定义但未在 SEOIR 中使用（可能用于其他用途）。
    """
    
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

    
# ==================== 上下采样模块 ====================
# Resizing modules
class Downsample(nn.Module):
    """Downsample spatial resolution by 2x and double channels."""
    """下采样模块：空间分辨率减半，通道数减半（通过 PixelUnshuffle 实现）。
       输入: (b, n_feat, h, w)
       输出: (b, n_feat//2, h/2, w/2)
       具体：先 3x3 卷积将通道数减少一半，然后 PixelUnshuffle(2) 将空间降采样 2 倍。
    """
    
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 将 (b, C, H, W) 变为 (b, C*4, H/2, W/2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsample spatial resolution by 2x and halve channels."""
    """上采样模块：空间分辨率加倍，通道数加倍（通过 PixelShuffle 实现）。
       输入: (b, n_feat, h, w)
       输出: (b, n_feat*2, h*2, w*2)
       具体：先 3x3 卷积将通道数加倍，然后 PixelShuffle(2) 将通道重新排列到空间。
    """
    
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # 将 (b, C*4, H, W) 变为 (b, C, H*2, W*2)
        )

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    """Transformer 块：包含 LayerNorm + Attention + FFN，均带有残差连接。
       这是构建 SEOIR 的基本单元。
    """
    
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, dw=None):
        # 先 norm，然后 attention，残差连接
        x = x + self.attn(self.norm1(x), dw)
        # 再 norm，然后 FFN，残差连接
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv."""
    """重叠的块嵌入：使用 3x3 卷积（步长1，padding1）将输入图像映射为特征图。
       输入: (b, in_c, H, W)
       输出: (b, embed_dim, H, W)  分辨率不变。
    """
    
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ==================== SEOIR 主干网络 ====================
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
    
    """
    SEOIR：用于图像复原的 U-Net 风格网络。
    特点：
        - 使用 TransformerBlock 作为基础构件
        - 编码器：4 个阶段，每个阶段有不同数量的块，分辨率逐级减半，通道数加倍
        - 瓶颈：最深层，处理全局信息
        - 解码器：对称上采样，通过跳跃连接融合编码器特征
        - 最后经过若干 refinement 块，输出与输入残差连接

    参数：
        inp_channels (int): 输入图像通道数，默认 3
        out_channels (int): 输出图像通道数，默认 3
        dim (int): 基础特征通道数，默认 28
        num_blocks (list): 每个编码器阶段（共4级）的 TransformerBlock 数量，默认 [3,5,5,7]
        num_refinement_blocks (int): 解码器后额外 refinement 块的数量，默认 4
        heads (list): 每个阶段的注意力头数，默认 [1,2,4,8]
        ffn_expansion_factor (float): FFN 中的扩展因子，默认 2
        bias (bool): 卷积中是否使用偏置，默认 False
        LayerNorm_type (str): 层归一化类型，默认 WITH_BIAS
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

        # 每个阶段的通道数
        d1, d2, d3, d4 = dim, dim * 2, dim * 4, dim * 8

        # 辅助函数：构建一个阶段（多个 TransformerBlock 的容器，支持逐块传 dw）
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

        # ---- 初始特征提取 ----
        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ---- 编码器（Encoder）----
        # Encoder path
        self.encoder_level1 = make_stage(d1, heads[0], num_blocks[0])
        
        # 下采样 + 第二级
        # Downsampling and level 2 processing
        self.down1_2 = Downsample(d1)
        self.encoder_level2 = make_stage(d2, heads[1], num_blocks[1])
        
        # 下采样 + 第三级
        # Downsampling and level 3 processing
        self.down2_3 = Downsample(d2)
        self.encoder_level3 = make_stage(d3, heads[2], num_blocks[2])
        
        # 下采样 + 瓶颈（最深层）
        # Downsampling and bottleneck (latent) processing
        self.down3_4 = Downsample(d3)
        self.latent = make_stage(d4, heads[3], num_blocks[3])
        
        # ---- 解码器（Decoder）----
        # 从 level4 到 level3：上采样 + 特征融合 + 处理
        # Decoder path
        # Level 4 to 3
        self.up4_3 = Upsample(d3)  # 将 d4 上采样到 d3 的分辨率（但通道数变化？实际 Upsample 输出通道数 d3*2，需要后续处理）
        self.reduce_chan_level3 = nn.Conv2d(d2 + d3, d3, kernel_size=1, bias=bias)  # 融合 skip 和上采样特征后降维
        self.reduce_dim_level3 = nn.Conv2d(d4, d3, kernel_size=1, bias=bias)  # 将瓶颈特征降维到 d3
        self.decoder_level3 = make_stage(d3, heads[2], num_blocks[2])
        
        # 从 level3 到 level2
        # Level 3 to 2
        self.up3_2 = Upsample(d3)
        self.reduce_chan_level2 = nn.Conv2d(d3, d2, kernel_size=1, bias=bias)
        self.decoder_level2 = make_stage(d2, heads[1], num_blocks[1])
        
        # 从 level2 到 level1
        # Level 2 to 1
        self.up2_1 = Upsample(d2)
        self.decoder_level1 = make_stage(d2, heads[0], num_blocks[0])
        
        # ---- 最终精炼和输出 ----
        # Final refinement and output
        self.refinement = make_stage(d2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(d2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # ---- 退化分类器 ----
        # SEOIR.__init__ 末尾（task_num=5 对应5退化任务）
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
        
        """
        前向传播：
            inp_img: (B, 3, H, W)  输入图像
            noise_emb: 可选噪声嵌入（此实现未使用）
        返回:
            output: (B, 3, H, W)  复原图像（残差连接到输入）
        """
        
        # ---- 初始特征提取 ----
        # --- Initial feature extraction
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, dim, H, W]

        # 辅助函数：逐块遍历 ModuleList，可选传入 dw
        def run_stage(x, stage, dw=None):
            for blk in stage:
                x = blk(x, dw)
            return x

        # ---- 编码器（下采样路径）----
        # --- Encoder path (downsampling)，encoder 阶段 dw 尚未计算，传 None
        out_enc_level1 = run_stage(inp_enc_level1, self.encoder_level1)  # [B, dim, H, W]

        inp_enc_level2 = self.down1_2(out_enc_level1)  # [B, dim*2, H/2, W/2]
        out_enc_level2 = run_stage(inp_enc_level2, self.encoder_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # [B, dim*4, H/4, W/4]
        out_enc_level3 = run_stage(inp_enc_level3, self.encoder_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)  # [B, dim*8, H/8, W/8]
        dw = self.deg_classifier(inp_enc_level4)       # [B, 5]
        latent = run_stage(inp_enc_level4, self.latent, dw=dw)  # [B, dim*8, H/8, W/8]

        # ---- 准备瓶颈特征用于解码器（将通道数从 dim*8 降为 dim*4）----
        # --- Prepare bottleneck features for decoder
        latent = self.reduce_dim_level3(latent)  # [B, dim*4, H/8, W/8]

        # ---- 解码器（上采样路径）----
        # --- Decoder path (upsampling with skip connections)，全部传 dw
        # Level 4 to 3
        inp_dec_level3 = self.up4_3(latent)  # [B, dim*4, H/4, W/4]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)  # 拼接上采样特征和 skip 连接
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # 降维到 d3
        out_dec_level3 = run_stage(inp_dec_level3, self.decoder_level3, dw=dw)

        # Level 3 to 2
        inp_dec_level2 = self.up3_2(out_dec_level3)  # [B, dim*2, H/2, W/2]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # 降维到 d2
        out_dec_level2 = run_stage(inp_dec_level2, self.decoder_level2, dw=dw)

        # Level 2 to 1
        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, dim*2, H, W]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)  # 拼接
        out_dec_level1 = run_stage(inp_dec_level1, self.decoder_level1, dw=dw)

        # ---- 最终精炼和输出 ----
        # --- Final refinement
        out_dec_level1 = run_stage(out_dec_level1, self.refinement, dw=dw)

        # 残差连接：输出 = 卷积结果 + 输入图像
        # Output with residual connection to input
        output = self.output(out_dec_level1) + inp_img

        return output

    
# ==================== 测试代码 ====================
# Calculating Parameters, FLOPs, Memory
if __name__ == "__main__":
    # 创建模型实例（使用默认参数）
    # Create model instance
    model = SEOIR(
        num_blocks=[3, 5, 5, 7],
        dim=28,
        ffn_expansion_factor=2,
        num_refinement_blocks=4
    ).cuda()
    
    # 创建测试输入 (1, 3, 224, 224)
    # Create test input
    x = torch.randn(1, 3, 224, 224).cuda()
    
    # 打印模型结构
    # Print model architecture
    print(model)
    
    # 前向传播（eval 模式下 forward 只返回 output，不返回 logits）
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    # 打印显存占用
    # Print memory usage
    print('{:>16s} : {:<.3f} [M]'.format(
        'Max Memory', 
        torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2
    ))
    print(f'Output shape: {out.shape}')
    
    # 计算并打印 FLOPs 和参数量
    # Calculate and print FLOPS and parameters
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))
