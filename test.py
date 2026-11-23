# 导入操作系统相关功能，用于路径操作、环境变量等
import os
# 导入命令行参数解析模块
import argparse
# 导入子进程管理模块（虽然本文件中未直接使用，但保留以备潜在需要）
import subprocess
# 导入路径处理模块，提供面向对象的文件系统路径操作
from pathlib import Path

# 数据处理相关库
# Data processing
import numpy as np
# 进度条库，用于显示循环进度
from tqdm import tqdm

# PyTorch 核心库
# PyTorch imports
import torch
import torch.nn as nn
# 数据加载器，用于批量加载数据
from torch.utils.data import DataLoader
# PyTorch Lightning 库，简化训练流程（此处用于定义 LightningModule）
import lightning.pytorch as pl

# Third-party evaluation metrics
# 第三方评估指标：LPIPS（学习感知图像块相似度）
import lpips

# Local imports
# 本地导入：SEOIR 模型定义（网络结构）
from net.seoir import SEOIR
# 数据集工具：用于不同任务的数据集类
from utils.dataset_utils import AnyDnTestDataset, SEOIRTestDataset, CDD11
# 验证工具：平均计量器、计算 PSNR 和 SSIM 的函数
from utils.val_utils import AverageMeter, compute_psnr_ssim
# 图像输入输出工具：保存张量格式的图像
from utils.image_io import save_image_tensor

# Set environment variables
# 设置环境变量，避免 NCCL（NVIDIA 集合通信库）异步错误
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# 设置 Hydra（配置管理库）错误显示为完整信息
os.environ["HYDRA_FULL_ERROR"] = "1"

# Remove SLURM env variables to avoid issues with Lightning
# 移除 SLURM 环境变量，避免与 Lightning 发生冲突
# SLURM 是集群作业调度系统，这些变量可能干扰 Lightning 的分布式训练设置
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]


# 定义 SEOIR 模型的 LightningModule 封装类
# 该类将模型、损失函数、优化器和学习率调度器整合在一起，方便 PyTorch Lightning 使用
class SEOIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 实例化 SEOIR 网络（具体结构在 net.seoir 中定义）
        self.net = SEOIR()
        # 使用 L1 损失函数（平均绝对误差）作为训练损失
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        # 前向传播：直接调用网络
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # 训练步骤函数，Lightning 会自动调用
        # batch 是一个元组，包含 (图像名列表, 退化图像张量, 干净图像张量)
        # 此处只使用退化图像和干净图像，de_id 被忽略
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        # 前向传播得到恢复图像
        restored = self.net(degrad_patch)
        # 计算 L1 损失
        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        # 记录训练损失（自动记录到 TensorBoard 等日志中）
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        # 学习率调度器步骤函数，在每个 epoch 后调用
        # 这里简单地将调度器步进一个 epoch，并获取当前学习率（但未使用）
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
        # 注意：此函数实际上未使用 lr 变量，仅为示例或未来扩展
    
    def configure_optimizers(self):
        # 配置优化器和学习率调度器
        # 使用 AdamW 优化器，学习率 2e-4
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        # 使用带线性预热和余弦退火的学习率调度器
        # warmup_epochs=15, max_epochs=150 表示前15个epoch线性增加学习率，之后余弦衰减
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)
        # 返回优化器和调度器（Lightning 会自动处理）
        return [optimizer],[scheduler]


# 定义 SEOIR 测试类，用于在多个任务上评估模型
class SEOIRTester:
    """Testing framework for SEOIR model."""
    
    def __init__(self, opt):
        """Initialize tester with options."""
        """初始化测试器，接收配置选项 opt"""
        self.opt = opt
        # 确定设备：如果 CUDA 可用则使用指定的 GPU，否则使用 CPU
        self.device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
        
        # Load model
        # 加载模型（从检查点）
        self.model = self._load_model()
        
        # 仅在请求时初始化 LPIPS（默认不启用，因为计算较慢）
        # Initialize LPIPS only if requested
        self.loss_fn_lpips = None
        if hasattr(opt, 'use_lpips') and opt.use_lpips:
            print("Initializing LPIPS for perceptual metrics...")
            # 使用 AlexNet 作为特征提取网络
            self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
            self.loss_fn_lpips.eval()
        
        # Create output directory
        # 创建输出目录（如果不存在）
        Path(opt.output_path).mkdir(exist_ok=True, parents=True)
    
        # 存储所有指标，按任务类型分类，后续用于计算平均值
        # Store all metrics for averaging
        self.all_metrics = {
            'denoise': [],
            'derain': [],
            'dehaze': [],
            'deblur': [],
            'enhance': []
        }

    def _load_model(self):
        """Load the SEOIR model from checkpoint."""
        """从检查点加载 SEOIR 模型"""
        
        '''
        print(f"Loading checkpoint: {self.opt.ckpt_path}")
        # load_from_checkpoint 会自动实例化 SEOIRModel 并加载权重
        model = SEOIRModel().load_from_checkpoint(self.opt.ckpt_path).to(self.device)
        # 设置为评估模式（禁用 dropout 等）
        model.eval()
        return model
        '''
        
        print(f"Loading checkpoint: {self.opt.ckpt_path}")
    
        # 1. 先实例化一个和训练时结构一致的模型（只是为了让 Lightning 解析 checkpoint）
        #    或者直接手动加载 state_dict，只加载 net 部分。
        model = SEOIRModel()  # 简单的 SEOIRModel，网络结构不变
        checkpoint = torch.load(self.opt.ckpt_path, map_location=self.device)

        # 2. 提取 net. 开头的参数
        net_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                          if k.startswith('net.')}
        
        # 2.5 key 重映射：训练版 lam_0/lam_2（独立属性） → 0901版 lam_net.0/lam_net.2（nn.Sequential）
        #     两者计算逻辑等价（Conv→GELU→Conv→Sigmoid），只是 PyTorch state_dict 命名规则不同：
        #     独立属性 self.lam_0 产生 key “lam_0”，nn.Sequential 产生 key “lam_net.0”
        #     形状完全一致，可安全映射
        remapped = {}
        for k, v in net_state_dict.items():
            new_k = k
            if '.lam_0.' in k:
                new_k = k.replace('.lam_0.', '.lam_net.0.')
            elif '.lam_2.' in k:
                new_k = k.replace('.lam_2.', '.lam_net.2.')
            remapped[new_k] = v
        net_state_dict = remapped

        # 3. 加载到 model.net 中
        # model.net.load_state_dict(net_state_dict)
        #    用 strict=False 避免其他潜在不匹配阻断加载；如有 missing/unexpected 会打印
        missing, unexpected = model.net.load_state_dict(net_state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")
            if missing: print(f"  missing[:5]: {missing[:5]}")
            if unexpected: print(f"  unexpected[:5]: {unexpected[:5]}")

        model = model.to(self.device)
        model.eval()
        return model
        
    
    def test_denoise(self, dataset, sigma):
        """Test denoising with specified noise level."""
        """测试去噪任务，sigma 为高斯噪声标准差"""
        # 构建输出路径：output_path/denoise/sigma/
        output_path = os.path.join(self.opt.output_path, f'denoise/{sigma}/')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        
        # 设置数据集的噪声水平（数据集内部会根据 sigma 添加噪声）
        dataset.set_sigma(sigma)
        # 创建 DataLoader，批量大小为1，不打乱数据，不使用多进程（num_workers=0 避免复杂情况）
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        # 评估数据集，返回指标字典（包含 psnr, ssim，可能还有 lpips）
        metrics = self._evaluate_dataset(testloader, output_path)
        
        # 将指标存入 all_metrics 的 denoise 列表
        # Store metrics for later averaging
        self.all_metrics['denoise'].append(metrics)
        
        # 构建输出字符串并打印
        # Build result string based on available metrics
        result_str = f"Denoising (sigma={sigma}): PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
        if 'lpips' in metrics:
            result_str += f", LPIPS: {metrics['lpips']:.4f}"
        print(result_str)
        
        return metrics
    
    def test_restoration(self, dataset, task):
        """Test other restoration tasks (derain, dehaze, deblur, enhance)."""
        """测试其他恢复任务（去雨、去雾、去模糊、低光增强）"""
        # 构建输出路径：output_path/task/
        output_path = os.path.join(self.opt.output_path, f'{task}/')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        
        # 设置数据集的任务类型（数据集内部会根据任务加载不同的图像对）
        dataset.set_dataset(task)
        # 创建 DataLoader
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        # 评估数据集
        metrics = self._evaluate_dataset(testloader, output_path)
        
        # 存储指标
        # Store metrics for later averaging
        self.all_metrics[task].append(metrics)

        # 打印结果
        # Build result string based on available metrics
        result_str = f"{task.capitalize()}: PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
        if 'lpips' in metrics:
            result_str += f", LPIPS: {metrics['lpips']:.4f}"
        print(result_str)
        
        return metrics
    
    def _evaluate_dataset(self, dataloader, output_path=None):
        """Evaluate model on dataset and optionally save results."""
        """
        核心评估函数：遍历 dataloader 中的所有样本，计算 PSNR、SSIM 和可选的 LPIPS。
        如果提供 output_path，则保存恢复后的图像。
        返回平均指标字典。
        """
        # 创建计量器，用于累加和平均
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        lpips_meter = AverageMeter() if self.loss_fn_lpips else None
    
        # 禁用梯度计算，加快推理并节省内存
        with torch.no_grad():
            # 使用 tqdm 显示进度条
            for (names, degrad_patch, clean_patch) in tqdm(dataloader):
                # 提取图像名称（可能是一个列表，取第一个元素）
                # Extract image name from list
                name = names[0] if isinstance(names[0], str) else names[0][0]
                
                # 将数据移动到指定设备
                # Move data to device
                degrad_patch = degrad_patch.to(self.device)
                clean_patch = clean_patch.to(self.device)

                # 模型推理
                # Run model inference
                restored = self.model(degrad_patch)
                
                # 计算 PSNR 和 SSIM，函数返回 (psnr, ssim, 有效像素数 N)
                # Calculate metrics
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)
                
                # 如果需要 LPIPS，则将图像值域从 [0,1] 映射到 [-1,1]（LPIPS 期望的输入范围）
                # Calculate LPIPS only if enabled
                if self.loss_fn_lpips:
                    restored_lpips = restored.clamp(0, 1) * 2 - 1
                    clean_lpips = clean_patch.clamp(0, 1) * 2 - 1
                    lpips_value = self.loss_fn_lpips(restored_lpips, clean_lpips).mean().item()
                    lpips_meter.update(lpips_value, N)
                
                # 如果需要保存图像（注释掉了，因为作者可能不需要保存所有输出）
                # Save output image if path provided
                if output_path is not None:
                    save_image_tensor(restored, os.path.join(output_path, f"{name}.png"))
        
    # 构建返回的指标字典
    # Build metrics dictionary conditionally
        metrics = {
            'psnr': psnr_meter.avg,
            'ssim': ssim_meter.avg,
        }
        
        if self.loss_fn_lpips:
            metrics['lpips'] = lpips_meter.avg

        return metrics
    
    def _calculate_average_metrics(self):
        """Calculate average metrics across all degradation types."""
        """计算所有已测试任务的平均指标（PSNR、SSIM、LPIPS）"""
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        # 遍历 all_metrics 字典中的每个任务列表
        # Collect all metrics
        for degradation_type, metrics_list in self.all_metrics.items():
            if metrics_list:  # Only process degradation types that were tested  仅处理有数据的任务
                for metrics in metrics_list:
                    all_psnr.append(metrics['psnr'])
                    all_ssim.append(metrics['ssim'])
                    if 'lpips' in metrics:
                        all_lpips.append(metrics['lpips'])
        
        # 计算平均值
        # Calculate averages
        avg_metrics = {}
        if all_psnr:
            avg_metrics['psnr'] = sum(all_psnr) / len(all_psnr)
        if all_ssim:
            avg_metrics['ssim'] = sum(all_ssim) / len(all_ssim)
        if all_lpips and len(all_lpips) > 0:
            avg_metrics['lpips'] = sum(all_lpips) / len(all_lpips)
            
        return avg_metrics
    
    def _print_average_metrics(self):
        """Print the average metrics across all degradation types."""
        """打印所有任务的平均指标"""
        avg_metrics = self._calculate_average_metrics()
        
        if not avg_metrics:
            return
            
        print("\n" + "="*50)
        print("AVERAGE RESULTS ACROSS ALL DEGRADATIONS:")
        print("-"*50)
        
        result_str = f"Average PSNR: {avg_metrics.get('psnr', 0):.2f}, "
        result_str += f"Average SSIM: {avg_metrics.get('ssim', 0):.4f}"
        
        if 'lpips' in avg_metrics:
            result_str += f", Average LPIPS: {avg_metrics.get('lpips', 0):.4f}"
            
        print(result_str)
        print("="*50)
        
        return avg_metrics

    def run_all_tests(self):
        """Run tests based on the specified mode."""
        """根据配置的 mode 运行相应的测试"""
        
        # mode 0 或 >=5 时测试去噪
        if self.opt.mode == 0 or self.opt.mode >= 5:
            print('<------ Testing Denoising: ------>')
            # 创建去噪数据集（这里只是准备数据集对象，具体在 _run_denoise_tests 中使用）
            # Create datasets
            denoise_datasets = self._create_denoise_datasets()
            self._run_denoise_tests(denoise_datasets)
        
        # mode 1 或 >=5 时测试去雨
        if self.opt.mode == 1 or self.opt.mode >= 5:
            print('<------ Testing Deraining: ------>')
            self._run_derain_tests()
            
        # mode 2 或 >=5 时测试去雾
        if self.opt.mode == 2 or self.opt.mode >= 5:
            print('<------ Testing Dehazing: ------>')
            self._run_dehaze_tests()
            
        # mode 3 或 >=6 时测试去模糊（注意 mode>=6 才测试去模糊和低光增强）
        if self.opt.mode == 3 or self.opt.mode >= 6:
            print('<------ Testing Deblurring: ------>')
            self._run_deblur_tests()
            
        # mode 4 或 >=6 时测试低光增强
        if self.opt.mode == 4 or self.opt.mode >= 6:
            print('<------ Testing Enhance: ------>')
            self._run_enhance_tests()
    
        # 最后打印所有任务的平均指标
        # Print average metrics at the end
        self._print_average_metrics()

    def _create_denoise_datasets(self):
        """Create datasets for denoising tests."""
        """创建去噪数据集列表，每个数据集对应一个测试子集（如 bsd68）"""
        denoise_datasets = []
        for split in self.opt.denoise_splits:
            split_path = os.path.join(self.opt.denoise_path, split)
            # AnyDnTestDataset 是去噪专用数据集，需要传入一个命名空间对象
            dataset = AnyDnTestDataset(argparse.Namespace(denoise_path=split_path))
            denoise_datasets.append((dataset, split))
        return denoise_datasets
    
    def _run_denoise_tests(self, denoise_datasets):
        """Run denoising tests for all datasets and noise levels."""
        """对所有去噪数据集和噪声水平进行测试"""
        for dataset, name in denoise_datasets:
            for sigma in [15, 25, 50]:
                print(f'Testing {name} with Sigma={sigma}...')
                self.test_denoise(dataset, sigma)
    
    def _run_derain_tests(self):
        """Run deraining tests."""
        """运行去雨测试"""
        for split in self.opt.derain_splits:
            print(f'Testing {split} rain streak removal...')
            split_path = os.path.join(self.opt.derain_path, split)
            derain_set = SEOIRTestDataset(
                argparse.Namespace(derain_path=split_path),
                task='derain',  # 指定任务类型
                addnoise=False,  # 不额外添加噪声
                sigma=15  # 噪声水平（虽然 addnoise=False，但参数仍传递）
            )
            self.test_restoration(derain_set, task="derain")
    
    def _run_dehaze_tests(self):
        """Run dehazing tests."""
        """运行去雾测试"""
        print('Testing SOTS outdoor dehazing...')
        # Reuse dataset from derain with different task
        split_path = os.path.join(self.opt.dehaze_path, self.opt.dehaze_splits[0])
        dehaze_set = SEOIRTestDataset(
            argparse.Namespace(dehaze_path=split_path),
            task='dehaze',
            addnoise=False,
            sigma=15
        )
        self.test_restoration(dehaze_set, task="dehaze")
    
    def _run_deblur_tests(self):
        """Run deblurring tests."""
        """运行去模糊测试（GOPRO 数据集）"""
        print('Testing Deblurring (GOPRO)...')
        for split in self.opt.deblur_splits:
            split_path = os.path.join(self.opt.gopro_path, split)
            deblur_set = SEOIRTestDataset(
                argparse.Namespace(gopro_path=split_path),
                task='deblur',
                addnoise=False,
                sigma=15,
            )
            self.test_restoration(deblur_set, task="deblur")
    
    def _run_enhance_tests(self):
        """Run low-light enhancement tests."""
        """运行低光增强测试（LOL 数据集）"""
        print('Testing Low-light Enhancement (LOL)...')
        for split in self.opt.enhance_splits:
            split_path = os.path.join(self.opt.enhance_path, split)
            enhance_set = SEOIRTestDataset(
                argparse.Namespace(enhance_path=split_path),
                task='enhance',
                addnoise=False,
                sigma=15,
            )
            self.test_restoration(enhance_set, task="enhance")

            
# 定义 CDD11 数据集的测试类（复合退化数据集）
class CDD11Tester:
    """Testing framework for CDD11 dataset using SEOIR model."""
    
    def __init__(self, opt):
        """Initialize tester with options and subset."""
        """初始化 CDD11 测试器"""
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
        
        # 从 trainset 名称中提取子集类型（例如 "CDD11_all" 提取出 "all"）
        # opt.trainset 格式如 "CDD11_all", "CDD11_single", "CDD11_double", "CDD11_triple"
        # Extract subset from trainset name
        _, self.subset = opt.trainset.split("_", maxsplit=1)
        
        # 加载模型
        # Load model
        self.model = self._load_model()
        
        # 可选 LPIPS
        # Initialize LPIPS only if requested
        self.loss_fn_lpips = None
        if hasattr(opt, 'use_lpips') and opt.use_lpips:
            print("Initializing LPIPS for perceptual metrics...")
            self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
            self.loss_fn_lpips.eval()
        
        # 创建输出目录：output_path/cdd11/subset/
        # Create output directory
        output_path = os.path.join(self.opt.output_path, f'cdd11/{self.subset}/')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        self.output_path = output_path
    
    def _load_model(self):
        """Load the SEOIR model from checkpoint."""
        """加载 SEOIR 模型检查点"""
        print(f"Loading checkpoint: {self.opt.ckpt_path}")
        model = SEOIRModel()  # test.py 中的 SEOIRModel，不需要 opt
        checkpoint = torch.load(self.opt.ckpt_path, map_location=self.device)

        net_state_dict = {
            k.replace('net.', ''): v
            for k, v in checkpoint['state_dict'].items()
            if k.startswith('net.')
        }

        # 加入 key 的兼容重映射（与 SEOIRTester 保持一致）
        remapped = {}
        for k, v in net_state_dict.items():
            new_k = k
            if '.lam_0.' in k:
                new_k = k.replace('.lam_0.', '.lam_net.0.')
            elif '.lam_2.' in k:
                new_k = k.replace('.lam_2.', '.lam_net.2.')
            remapped[new_k] = v
        net_state_dict = remapped

        # 使用 strict=False 避免因为部分多余的 key 直接报错
        missing, unexpected = model.net.load_state_dict(net_state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")
            if missing: print(f"  missing[:5]: {missing[:5]}")
            if unexpected: print(f"  unexpected[:5]: {unexpected[:5]}")

        model = model.to(self.device)
        model.eval()
        return model
    
    def run_test(self):
        """运行 CDD11 测试"""
        """Run tests on CDD11 dataset with the specified subset."""
        print(f'<------ Testing CDD11 ({self.subset}): ------>')
        
        # 创建 CDD11 数据集，split="test"，subset 指定测试子集（all/single/double/triple）
        # Create dataset
        dataset = CDD11(self.opt, split="test", subset=self.subset)
        
        # 创建 DataLoader
        # Create dataloader
        testloader = DataLoader(
            dataset, 
            batch_size=1, 
            pin_memory=True, 
            shuffle=False, 
            num_workers=0
        )
        
        # 评估数据集
        # Evaluate on dataset
        metrics = self._evaluate_dataset(testloader, self.output_path)
        
        # 打印结果
        # Print results
        result_str = f"CDD11 ({self.subset}): PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
        if 'lpips' in metrics:
            result_str += f", LPIPS: {metrics['lpips']:.4f}"
        print(result_str)
        
        return metrics
    
    def _evaluate_dataset(self, dataloader, output_path=None):
        """Evaluate model on dataset and optionally save results."""
        """
        评估 CDD11 数据集（与 SEOIRTester 中的 _evaluate_dataset 类似，但增加了图像保存功能）
        """
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        lpips_meter = AverageMeter() if self.loss_fn_lpips else None
    
        with torch.no_grad():
            for (names, degrad_patch, clean_patch) in tqdm(dataloader):
                # Extract image name from list
                # 提取图像名称
                name = names[0] if isinstance(names[0], str) else names[0][0]
                
                # Move data to device
                # 数据移动到设备
                degrad_patch = degrad_patch.to(self.device)
                clean_patch = clean_patch.to(self.device)

                # Run model inference
                # 推理
                restored = self.model(degrad_patch)
                
                # Calculate metrics
                # 计算 PSNR, SSIM
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)
                
                # Calculate LPIPS only if enabled
                # 可选 LPIPS
                if self.loss_fn_lpips:
                    restored_lpips = restored.clamp(0, 1) * 2 - 1
                    clean_lpips = clean_patch.clamp(0, 1) * 2 - 1
                    lpips_value = self.loss_fn_lpips(restored_lpips, clean_lpips).mean().item()
                    lpips_meter.update(lpips_value, N)
                
                # Save output image if path provided
                # 保存输出图像（如果指定了输出路径）
                if output_path:
                    save_image_tensor(restored, os.path.join(output_path, f"{name}.png"))
        
        # Build metrics dictionary conditionally
        # 构建指标字典
        metrics = {
            'psnr': psnr_meter.avg,
            'ssim': ssim_meter.avg,
        }
        
        if self.loss_fn_lpips:
            metrics['lpips'] = lpips_meter.avg

        return metrics


def parse_args():
    """Parse command line arguments."""
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SEOIR testing script")
    
    # Testing mode
    # 测试模式：0=仅去噪，1=仅去雨，2=仅去雾，3=仅去模糊，4=仅增强，5=前三项（去噪去雨去雾），6=全部五项
    parser.add_argument('--mode', type=int, default=6,
                       help='Test mode: 0=denoise, 1=derain, 2=dehaze, 3=deblur, '
                            '4=enhance, 5=three tasks, 6=all five tasks')
    
    # 训练集名称，用于区分测试类型：SEOIR 表示常规测试，CDD11_xxx 表示复合退化测试
    parser.add_argument('--trainset', default="SEOIR", 
                        help=["SEOIR", "CDD11_all", "CDD11_single", "CDD11_double", "CDD11_triple"])


    # 设备设置
    # Device settings
    parser.add_argument('--cuda', type=int, default=0,
                       help='CUDA device index')
    
    # 数据集路径
    # Dataset paths
    parser.add_argument('--denoise_path', type=str, 
                       default="/dataset/low-level/test/denoise/",
                       help='Path to denoising test images')
    parser.add_argument('--derain_path', type=str,
                       default="/dataset/low-level/test/derain/",
                       help='Path to deraining test images')
    parser.add_argument('--dehaze_path', type=str,
                       default="/dataset/low-level/test/dehaze/",
                       help='Path to dehazing test images')
    parser.add_argument('--gopro_path', type=str,
                       default="/datasets/low-level/test/deblur/",
                       help='Path to deblurring test images')
    parser.add_argument('--enhance_path', type=str,
                       default="/datasets/low-level/test/enhance/",
                       help='Path to low-light enhancement test images')
    parser.add_argument('--cdd11_path', type=str, 
                       default="/datasets/low-level/cdd11/",
                       help='Path to CDD11 test images')

    # 输出设置
    # Output settings
    parser.add_argument('--output_path', type=str,
                       default="output/",
                       help='Output directory for restored images')
    parser.add_argument('--ckpt_name', type=str,
                       default="model.ckpt",
                       help='Checkpoint filename')
    
    # 是否计算 LPIPS（默认不计算，因为较慢）
    parser.add_argument('--use_lpips', action='store_true',
                       help='Enable LPIPS metric calculation (slower but more perceptually accurate)')

    args = parser.parse_args()
    
    # Set additional parameters
    # 根据参数构造检查点完整路径（假设检查点保存在 train_ckpt 目录下）
    args.ckpt_path = os.path.join("train_ckpt", args.ckpt_name)
    # 定义各个任务的数据子集名称（这些子目录存在于对应的路径下）
    args.denoise_splits = ["bsd68/"]
    args.derain_splits = ["Rain100L/"]
    args.deblur_splits = ["gopro/"]
    args.enhance_splits = ["lol/"]
    args.dehaze_splits = ["SOTS/"]
    
    return args

def main():
    """Main function."""
    """主函数"""
    # Parse arguments
    # 解析参数
    args = parse_args()
    
    # Set random seeds for reproducibility
    # 设置随机种子，保证结果可重复
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Set CUDA device
    # 设置 CUDA 设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
    
    # 根据 trainset 参数决定使用哪种测试器
    if "CDD11" in args.trainset:
        # Create tester and run tests for CDD11
        # 复合退化测试
        tester = CDD11Tester(args)
        tester.run_test()

    elif "SEOIR" in args.trainset:
        # Create tester and run tests
        # 标准多任务测试
        tester = SEOIRTester(args)
        tester.run_all_tests()
    else:
        raise ValueError(f"Unknown dataset: {args.trainset}")

# 脚本入口点
if __name__ == '__main__':
    main()