#!/bin/bash

# Ensure the script is executable: chmod +x run_ablation.sh

log_lrs=(-15 -16 -13)
layer_types=("plain" "batchnorm" "decorrelation")

# log_lrs=(-8)
# layer_types=("decorrelation")

for log_lr in "${log_lrs[@]}"
do
    for layer in "${layer_types[@]}"
    do
        lr=$(python -c "print(2 ** $log_lr)")
        run_name="3layer_${layer}_lr_${lr}"
        echo "Running with layer type: $layer, learning rate: $lr"
        python run.py \
            --layer-type=$layer \
            --learning-rate=$lr \
            --wandb-project="cifar10_ablation" \
            --wandb-run-name=$run_name \
            --num-epochs=10 \
            --weight-decay=0.0005
    done
done
