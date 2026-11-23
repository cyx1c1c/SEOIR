import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)



class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, epsilon=1e-6, alpha=0.1):
        """
        Focal L1 Loss with adjusted weighting for output values in [0, 1].
        
        Args:
            gamma (float): Focusing parameter. Larger gamma focuses more on hard examples.
            epsilon (float): Small constant to prevent weights from being zero.
            alpha (float): Scaling factor to normalize error values.
        """
        super(FocalL1Loss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha  # Scaling factor to prevent error values from being too small.

    def forward(self, pred, target):
        """
        Compute the Focal L1 Loss between the predicted and target images.
        
        Args:
            pred (torch.Tensor): Predicted image [b, c, h, w].
            target (torch.Tensor): Ground truth image [b, c, h, w].
        
        Returns:
            torch.Tensor: Scalar Focal L1 Loss.
        """
        # Compute the absolute error (L1 Loss) and scale it by alpha
        abs_err = torch.abs(pred - target) / self.alpha
        
        # Apply a logarithmic transformation to the error to prevent very small weights
        focal_weight = (torch.log(1 + abs_err + self.epsilon)) ** self.gamma
        
        # Compute the weighted loss
        focal_l1_loss = focal_weight * abs_err
        
        # Return the mean loss across all pixels
        return focal_l1_loss.mean()
    
    
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    
    
class TemperatureScheduler:
    def __init__(self, start_temp, end_temp, total_steps):
        """
        Scheduler for Gumbel-Softmax temperature that decreases using a cosine annealing schedule.

        Args:
        - start_temp (float): Initial temperature (e.g., 5.0).
        - end_temp (float): Final temperature (e.g., 0.01).
        - total_steps (int): Total number of steps/epochs to anneal over.
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps

    def get_temperature(self, step):
        """
        Get the temperature value for the current step, following a cosine annealing schedule.
        
        Args:
        - step (int): Current step or epoch.
        
        Returns:
        - temperature (float): The temperature for the Gumbel-Softmax at this step.
        """
        if step >= self.total_steps:
            return self.end_temp
        
        # Cosine annealing formula to compute the temperature
        cos_inner = math.pi * step / self.total_steps
        #temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + math.cos(cos_inner))
        temp = self.start_temp + 0.5 * (self.end_temp - self.start_temp) * (1 - math.cos(cos_inner))
        
        return temp


class MS_SSIMLoss(nn.Module):  # increase by cyx
    """
    MS-SSIM 损失封装：计算 1 - MS-SSIM，值越小表示重建质量越高
    用于与 L1 损失加权组合，作为感知损失项
    """
    def __init__(self, data_range=1.0):
        super(MS_SSIMLoss, self).__init__()
        # 初始化 torchmetrics 的 MS-SSIM 计算器，设置数据范围为 [0,1]，多尺度权重为等权
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=5,
            betas=(0.5, 0.5, 0.5, 0.5, 0.5)
        )
    def forward(self, pred, target):
        # 返回 1 - MS-SSIM，与 L1 损失同向优化（越小越好）
        return 1 - self.ms_ssim(pred, target)

    
class HIGH_FREQ(nn.Module):
    """
    高频损失函数：计算高频残差的 L1 损失
    用于与 L1 损失加权组合，作为感知损失项
    """
    def __init__(self):
        super().__init__()
        self.laplacian = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.laplacian.weight.data = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]]).float().repeat(3,1,1,1)
        for p in self.laplacian.parameters():
            p.requires_grad = False
 
    def forward(self, restored, clean):
        # return F.l1_loss(x, torch.zeros_like(x), reduction='mean')  # 简化，实际可用 laplacian
        high_restored = self.laplacian(restored)
        high_clean = self.laplacian(clean)
        return F.l1_loss(high_restored, high_clean, reduction='mean')

    
class GRAD_DIFF(nn.Module):
    """
    梯度损失函数：计算梯度残差的 L1 损失
    用于与 L1 损失加权组合，作为感知损失项
    """
    def __init__(self):
        super(GRAD_DIFF, self).__init__()
 
    def forward(self, x, y):
        C = x.shape[1]  # 获取输入通道数
        sobel_x = torch.tensor([[1, 0, -1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        # 为每个通道复制 sobel 核
        sobel_x = sobel_x.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
        grad_x_r = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_x_c = F.conv2d(y, sobel_x, padding=1 ,groups=C)
        return F.l1_loss(grad_x_r, grad_x_c, reduction='mean')