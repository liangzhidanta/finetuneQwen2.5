#!/bin/bash
# run_models.sh
# 本脚本分别训练两个模型（均使用 MSELoss 作为损失函数），
# 模型之间的区别在于加载不同的预训练编码器权重。
# 训练结束后，分别调用 evaluate_MultiModalEncoderDecoderModel.py 对两个模型进行评估。
#
# 使用方法：
#   chmod +x run_models.sh
#   ./run_models.sh

# 使用CUDA:1
export CUDA_VISIBLE_DEVICES=1

###############################
# 公共参数设置
###############################

LOSS="HYBRID"
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-4
PATIENCE=5

# -------------------- 添加恢复参数 --------------------
# 模型1恢复路径（可选，示例：RESUME_PATH1="./checkpoints/epoch_15.pth"）
RESUME_PATH1=""
# 模型2恢复路径（可选）
RESUME_PATH2=""
# ------------------------------------------------------

###############################
# 模型1相关参数设置（加载预训练编码器1）
###############################
CHECKPOINT_DIR1="./fig1/wirelessbind_s8"
PRETRAINED_ENCODER_PATH1="/data2/wzj/BeamMM_checkpoints/mmwave_gps_joint_s8.pth"  # MMWAVE ANCHOR

###############################
# 模型2相关参数设置（加载预训练编码器2）
###############################
CHECKPOINT_DIR2="./checkpoints_model2"
PRETRAINED_ENCODER_PATH2="/data2/wzj/BeamMM_checkpoints/mmwave_gps_joint-01-15-152626.pth"   # VISION ANCHOR

###############################
# 确保保存检查点的目录存在
###############################
mkdir -p "$CHECKPOINT_DIR1"
mkdir -p "$CHECKPOINT_DIR2"

###############################
# 开始训练模型1
###############################
echo "============================="

python -W ignore -u stride1-10.py \
    --loss "$LOSS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --patience "$PATIENCE" \
    --checkpoint_dir "$CHECKPOINT_DIR1" \
    --dataset_start_idx 1 \
    --dataset_end_idx 9 \
    --pretrained_encoder_path "$PRETRAINED_ENCODER_PATH1" 
    #--resume "$RESUME_PATH1" 
echo "时间测试完成"

echo "============================="
