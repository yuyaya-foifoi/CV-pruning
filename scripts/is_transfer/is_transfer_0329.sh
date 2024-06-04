#!/bin/bash

# パラメータを変数として定義
learning_rate=0.1
num_epochs=100
weight_decay=0.0001
seeds=3
momentum=0.9
batch_size=128
dataset_name="SVHN"
source_path="./logs/CIFAR10/is_prune/ensemble_output/seed_{}/2024_03_27_16_44_39/resnet_slth1_state.pkl"


# Pythonスクリプトを実行
poetry run python src/tools/is_transfer/train.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --batch_size $batch_size \
    --dataset_name $dataset_name \
    --source_path $source_path \

