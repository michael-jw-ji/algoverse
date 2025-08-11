#!/bin/bash

# List of checkpoint steps
checkpoints=(15000 23000 31000 39000 47000 55000 63000 71000 79000 87000 95000 103000 111000 119000 127000 135000 143000)

for step in "${checkpoints[@]}"; do
    echo "Running checkpoint $step..."
    python experiments/circuit_discovery_step.py \
        EleutherAI/pythia-70m \
        --checkpoint $step \
        --dataset math \
        --format few-shot \
        --format_params shots=3 \
        --seed 0 \
        results/gsm8k_3shot_seed0_step${step} \
        --batch_size 4
done
