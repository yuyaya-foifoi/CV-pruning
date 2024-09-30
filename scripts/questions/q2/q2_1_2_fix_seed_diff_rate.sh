#!/bin/bash

learning_rate=0.1
num_epochs=100
weight_decay=0.0001
seeds=5
momentum=0.9
n_class=10
batch_size=128
dataset_name="CIFAR10"
dir_name="20240630_q2_1_diff_rate"

remain_rate=0.125
echo "Running with remain_rate=$remain_rate"
poetry run python src/tools/is_prune/q2/train_ensemble_diff_score_fix_seed.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --n_class $n_class \
    --batch_size $batch_size \
    --dataset_name $dataset_name \
    --remain_rate $remain_rate \
    --dir_name $dir_name

remain_rate=0.10
echo "Running with remain_rate=$remain_rate"
poetry run python src/tools/is_prune/q2/train_ensemble_diff_score_fix_seed.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --n_class $n_class \
    --batch_size $batch_size \
    --dataset_name $dataset_name \
    --remain_rate $remain_rate \
    --dir_name $dir_name