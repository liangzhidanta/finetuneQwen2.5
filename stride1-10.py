# train_MultiModalEncoderDecoderModel.py

import torch
from data_loader_decoder import CustomDataset_decoder, collate_fn
from torchvision import transforms
import glob
import os
from collections import OrderedDict, defaultdict
import math
import csv
import random
from tqdm import tqdm  # 引入 tqdm 库
import time  # 引入 time 模块
import argparse  # 引入 argparse 模块
import sys
import numpy as np
sys.path.append('./imagebind_beam')
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from model import MultiModalEncoderDecoderModel
from decoder import MMWaveDecoder
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

# 设置seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 如果使用的是 cudnn，以下设置可以进一步提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 选择一个固定的随机种子

#------------------------------
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiModalEncoderDecoderModel with MSE or NMSE loss.')
    parser.add_argument('--loss', type=str, choices=['MSE', 'NMSE','TOPK','HYBRID','CE'], default='MSE',
                        help='选择损失函数类型：MSE、NMSE、TOPK、HYBRID、CE。默认是 MSE。')
    parser.add_argument('--epochs', type=int, default=50, help='训练的总轮数。默认是50。')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小。默认是16。')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率。默认是1e-4。')
    parser.add_argument('--patience', type=int, default=10, help='早停步数。')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型检查点保存目录。默认是./checkpoints。')
    parser.add_argument('--pretrained_encoder_path', type=str, required=True,
                        help='预训练编码器权重的路径。')
    parser.add_argument('--dataset_start_idx', type=int, default=1, help='数据集起始索引。默认是1。')
    parser.add_argument('--dataset_end_idx', type=int, default=9, help='数据集结束索引。默认是9。')
    parser.add_argument('--resume', type=str, default=None, help='路径 to 中断的模型检查点 (e.g., epoch_15.pth)')
    return parser.parse_args()
def main():
    # 解析命令行参数
    args = parse_args()

    # 1. 数据加载和划分部分

    # 定义转换，与预训练时保持一致
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 与预训练时一致
                             std=[0.229, 0.224, 0.225]),
    ])


    initial_output_length = 3  # 先随便给一个 value
    # 初始化 CustomDataset_decoder
    modal = 'mmwave_gps'  # 选择模态
    input_length = 8
    output_length = initial_output_length
    batch_size = args.batch_size  # 从命令行参数获取批量大小

    # 2. 模型加载部分

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")

    # 初始化编码器
    encoder = ImageBindModel(
        video_frames=input_length,  # 与数据加载时的 input_length 一致
        # 其他参数应与预训练时保持一致
    ).to(device)

    # 加载预训练的编码器权重
    model_save_path = args.pretrained_encoder_path  # 使用命令行参数提供的路径
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Pretrained encoder weights not found at {model_save_path}")

    state_dict = torch.load(model_save_path, map_location=device)
    print(f"Loaded state_dict from {model_save_path}")

    # 处理多GPU保存的模型（如果适用）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 加载权重到编码器
    encoder.load_state_dict(new_state_dict)
    print("Encoder weights loaded successfully.")

    # 设置编码器为冻结（如果不需要微调）
    for param in encoder.parameters():
        param.requires_grad = False

    # 初始化解码器

    decoder = MMWaveDecoder(
        embed_dim=768,        # 根据 decoder 的定义
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        dropout=0.1,
        output_dim=64,        # mmWave 数据的维度，根据实际情况调整
        max_seq_length=output_length
    ).to(device)

    # 初始化集成模型，提供各模态的特征维度
    vision_dim = 768  # 假设 VISION 特征维度为 768
    gps_dim = 768     # 假设 GPS 特征维度为 768
    mmwave_dim = 768  # 确保 mmwave_dim 设置为 768

    # 初始化集成模型
    model = MultiModalEncoderDecoderModel(
        encoder=encoder, 
        decoder=decoder, 
        vision_dim=vision_dim, 
        gps_dim=gps_dim, 
        mmwave_dim=mmwave_dim,
        input_length=input_length,
        output_length=output_length
    ).to(device)
    model.eval()

#-----------------------------------------------------------------
    print("测试推理时间随步长的变化:")
    scenario9_dataset_path = [f'/data2/wzj/Datasets/DeepSense/scenario{i}/' for i in range(9, 10)]  # scenario9
    scenario9_data_csv_paths = []
    for path in scenario9_dataset_path:
        scenario9_data_csv_paths.extend(glob.glob(os.path.join(path, '*.csv')))
    print(f"Found {len(scenario9_data_csv_paths)} CSV files for testing reasoning time.")
    print(f"dataset_path: {scenario9_dataset_path}")
    
    scenario9_dataset = CustomDataset_decoder(
        scenario9_data_csv_paths, 
        transform=default_transform, 
        modal=modal, 
        input_length=input_length, 
        output_length=output_length
    )
    # 随机选择100个样本的索引
    total_size = len(scenario9_dataset)
    random.seed(42)  # 设置随机种子确保可复现
    indices = list(range(total_size))
    selected_indices = random.sample(indices, 640)  # 随机抽取100个索引

    # 创建子集
    subset = Subset(scenario9_dataset, selected_indices)

    zeroshot_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print(f"测试集样本数: {len(scenario9_dataset)}")

#----------------------------------------------------------------

    def measure_forward_full_dataset(fn_forward, model, data_loader, device,
                                    num_warmup: int = 1, num_runs: int = 5):
        """
        对“遍历整个 data_loader，对每个 batch 都 run 一次 model(inputs, tgt_seq)”进行
        预热 + N 次正式测时。返回 mean_ms, std_ms，单位是毫秒。

        - fn_forward:       你包装好的 forward(model, data_loader, device) 函数
        - model:            已加载到 device，并且 eval() 的 nn.Module
        - data_loader:      DataLoader，batch_size 已设置好
        - device:           torch.device("cuda") 或 "cpu"
        - num_warmup:       预热遍历数据集的次数，默认 1 次
        - num_runs:         正式测时次数，默认 5 次（因为跑完整个数据集通常比较慢）

        返回：
        (mean_latency_ms, std_latency_ms)
        """
        # —— 在这里打印当前 model.output_length —— #
        print(f"当前 model.output_length = {model.output_length}, decoder.max_seq_length = {model.decoder.max_seq_length}")

        # —— Warm-up —— #
        for _ in range(num_warmup):
            fn_forward(model, data_loader, device)

        # —— 正式测时 —— #
        latencies = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            fn_forward(model, data_loader, device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)

        arr = np.array(latencies, dtype=np.float32)
        # 这里 len(data_loader) 就是“DataLoader 里 batch 的数量”
        num_batches = len(data_loader)
        mean_full_dataset = float(arr.mean())    # 整个数据集一次过的均耗时
        std_full_dataset  = float(arr.std())

        # 换算成“单个 batch 的平均耗时”：
        mean_per_batch = mean_full_dataset / num_batches
        std_per_batch  = std_full_dataset  / num_batches

        return mean_per_batch, std_per_batch
    # ---------------------------------------------------------
    def forward(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='forward', leave=False)):
                # 准备输入
                inputs = {}
                if modal == 'gps':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.GPS] = batch['gps'].to(device)
                elif modal == 'mmwave':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
                elif modal == 'mmwave_gps':
                    inputs[ModalityType.VISION] = batch['video'].to(device)
                    inputs[ModalityType.GPS] = batch['gps'].to(device)
                    inputs[ModalityType.MMWAVE] = batch['mmwave'].to(device)
                else:
                    raise ValueError(f"Unsupported modal: {modal}")

                target_mmwave = batch['target_mmwave'].to(device)  # [B, output_length, D]

                # 准备目标序列的输入
                tgt_seq = torch.zeros(target_mmwave.size(0), output_length, dtype=torch.long, device=device)  # [B, output_length]

                # 前向传播
                _ = model(inputs, tgt_seq)  # [B, output_length, D]
            
        return
    #---------------------------------------------------------

    results = {
        "Ours": {"mean": [], "std": []},  
    }

    num_warmup_runs = 1   # 预热遍历数据集的次数
    num_timed_runs   = 5  # 正式测时遍历数据集的次数（可以适当增大，但会更耗时）

    # 确保保存模型的目录存在
    os.makedirs(args.checkpoint_dir, exist_ok=True)  
    # 加载最佳模型
    best_model_path = os.path.join(args.checkpoint_dir, 'multimodal_encoder_decoder_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        print("Loaded best model for testing.")
    else:
        print(f"Best model not found at {best_model_path}. Skipping test evaluation.")
    print ("testing...")

    for k in range(1, 11):
        # 1) 动态修改全局 output_length
        output_length = k

        # 2) 把 k 传给两个模型，让它们内部知道“预测 k 步”
        #    - Model A 例子：QwenReprogPatchHeadLight 在 __init__ 里存了 output_length，
        #      并且 forward 里会读取它；所以这里改 output_length，。
        model.output_length = k
        model.decoder.max_seq_length = k  # 更新解码器的最大序列长度

        # —— 测 Model A 遍历整个数据集所用时间 —— #
        mean_ours, std_ours = measure_forward_full_dataset(
            fn_forward   = forward,
            model        = model,
            data_loader  = zeroshot_loader,
            device       = device,
            num_warmup   = num_warmup_runs,
            num_runs     = num_timed_runs
        )
        results["Ours"]["mean"].append(mean_ours)
        results["Ours"]["std"].append(std_ours)
        print(
            f"k={k:2d} | "
            f"Our Model: {mean_ours:.1f}±{std_ours:.1f} ms "
        )

    with open("inference_latency_per_batch.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "Ours_mean_ms", "Ours_std_ms"])
        for idx, k in enumerate(range(1, 11)):
            writer.writerow([
                k,
                results["Ours"]["mean"][idx],
                results["Ours"]["std"][idx]
            ])

    print("所有 k=1..10 的遍历数据集推理时延已保存到 inference_latency_per_batch.csv")

if __name__ == '__main__':
    main()
