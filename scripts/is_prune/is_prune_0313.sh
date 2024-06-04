#!/bin/bash

# パラメータを変数として定義
learning_rate=0.1
num_epochs=100
weight_decay=0.0001
seeds=3
momentum=0.9
batch_size=128

# Pythonスクリプトを実行
poetry run python src/tools/is_prune/train_w_ensemble_per_layer.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --batch_size $batch_size


# パラメータを変数として定義
learning_rate=0.1
num_epochs=100
weight_decay=0.0001
seeds=3
momentum=0.9
batch_size=128

# Pythonスクリプトを実行
poetry run python src/tools/is_prune/train_w_optimize_k.py \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --weight_decay $weight_decay \
    --seeds $seeds \
    --momentum $momentum \
    --batch_size $batch_size