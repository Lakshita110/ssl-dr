#!/bin/bash

# Ensure the script exits on any command failure
set -e

# Define paths and arguments
DATA_PATH="/cluster/tufts/cs152l3dclass/ljain01"
RESULT_DIR="./results"         # Directory to save results
ARCH="resnet18"               # Default architecture
BATCH_SIZE=10                # Batch size
EPOCHS=300                    # Total epochs
LEARNING_RATE=0.03            # Learning rate
WORKERS=2                     # Number of data loader workers
SEED_START=0                  # Starting seed
SEED_END=5                    # Ending seed
LOW_DIM=128                   # Feature dimension
NCE_K=4096                    # Number of negative samples for NCE
NCE_T=0.07                    # Temperature for softmax
NCE_M=0.5                     # Momentum for NCE
WEIGHT_DECAY=1e-4             # Weight decay
PRINT_FREQ=10                 # Print frequency
RESUME_CHECKPOINT=""          # Path to checkpoint file if resuming

# Additional flags (adjust as needed)
EVALUATE=""                   # Add `--evaluate` if you want to evaluate only
SYNTHESIS_FLAG=""             # Add `--synthesis` to enable synthesis
MULTIAUG_FLAG="--multitask --multiaug"              # Add `--multiaug` to enable multi-augmentation
DOMAIN_FLAG=""                # Add `--domain` to enable domain-based multitasking

# Construct the Python command
python3 kaggle_main.py \
    "$DATA_PATH" \
    --arch "$ARCH" \
    --workers "$WORKERS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --print-freq "$PRINT_FREQ" \
    --resume "$RESUME_CHECKPOINT" \
    --low-dim "$LOW_DIM" \
    --nce-k "$NCE_K" \
    --nce-t "$NCE_T" \
    --nce-m "$NCE_M" \
    --result "$RESULT_DIR" \
    --seedstart "$SEED_START" \
    --seedend "$SEED_END" \
    $EVALUATE \
    $SYNTHESIS_FLAG \
    $MULTIAUG_FLAG \
    $DOMAIN_FLAG