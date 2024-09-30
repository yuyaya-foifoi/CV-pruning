#!/bin/bash

learning_rate=0.01
num_epochs=100
weight_decay=0.0001
seeds=5
momentum=0.9
batch_size=128
dataset_name="CIFAR10"
dir_name="20240619_follow_up_pruning_big_model"
resnet_name="ResNet152"

remain_rate=0.1
echo "Running with remain_rate=$remain_rate"
poetry run python src/tools/is_prune/train.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --batch_size $batch_size \
    --dataset_name $dataset_name \
    --remain_rate $remain_rate \
    --dir_name $dir_name \
    --resnet_name $resnet_name \

remain_rate=0.06
echo "Running with remain_rate=$remain_rate"
poetry run python src/tools/is_prune/train.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --batch_size $batch_size \
    --dataset_name $dataset_name \
    --remain_rate $remain_rate \
    --dir_name $dir_name \
    --resnet_name $resnet_name \
    

remain_rate=0.02
echo "Running with remain_rate=$remain_rate"
poetry run python src/tools/is_prune/train.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --batch_size $batch_size \
    --dataset_name $dataset_name \
    --remain_rate $remain_rate \
    --dir_name $dir_name \
    --resnet_name $resnet_name \