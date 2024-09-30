#!/bin/bash

# パラメータを変数として定義
learning_rate=0.1
num_epochs=100
weight_decay=0.0001
seeds=5
momentum=0.9
batch_size=128
dataset_name="CIFAR10"
source_path="./logs/STL10/is_prune/baseline/ResNet18/20240627/remain_rate_30/seed_{}/2024_06_27_10_29_21/resnet_slth_state.pkl"
source_dataset_name="STL10"
dir_name="0627_stl10_to_cifar"

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
    --source_dataset_name $source_dataset_name


source_path="./logs/SVHN/is_prune/baseline/ResNet18/20240627/remain_rate_30/seed_{}/2024_06_27_11_47_18/resnet_slth_state.pkl"
source_dataset_name="SVHN"
dir_name="0627_svhn_to_cifar"

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
    --source_dataset_name $source_dataset_name
