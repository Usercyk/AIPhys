#!/bin/bash
# ================================
# Run PINN with Accelerate
# nohup ./train.sh > output.log 2>&1 &
# ================================

export CUDA_VISIBLE_DEVICES=5,6,7,8,9

CONFIG_PATH=/home/stu2400011486/assignments/assignment3/multi_gpu.yaml
TRAINER_PATH=/home/stu2400011486/assignments/assignment3/src/train.py

# 打印设备信息以便检查
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Accelerate config: $CONFIG_PATH"

IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#DEVICES[@]}
echo "Detected $NUM_GPUS visible GPU(s)"

# 启动训练（命令行指定 num_processes 会覆盖配置文件）
accelerate launch \
    --config_file "$CONFIG_PATH" \
    --num_processes "$NUM_GPUS" \
    "$TRAINER_PATH"