#!/bin/bash
#SBATCH --job-name=hyperparam-job        # Job name
#SBATCH --output=job_output_%A.txt       # Standard output and error log
#SBATCH --ntasks=1                       # Run a single task (one job)
#SBATCH --cpus-per-task=20                # Number of CPU cores per task
#SBATCH --mem=32G                        # Total memory
#SBATCH --time=10:00:00                  # Maximum execution time (hh:mm:ss)
#SBATCH -p hugheslab                  # Specify partition (if GPU is needed)
#SBATCH --gres=gpu:rtx_6000:1                   # Request one GPU (if needed)

# Ensure the script exits on any command failure
set -e

# Define paths and arguments
DATA_PATH="/cluster/tufts/cs152l3dclass/ljain01"
RESULT_DIR="./results1"         # Directory to save results
ARCH="resnet18"                # Default architecture
BATCH_SIZE=64                  # Batch size
EPOCHS=200                     # Total epochs
WORKERS=8                      # Number of data loader workers
SEED_START=0                   # Starting seed
SEED_END=5                     # Ending seed
LOW_DIM=128                    # Feature dimension
NCE_K=4096                     # Number of negative samples for NCE
NCE_T=0.07                     # Temperature for softmax
NCE_M=0.5                      # Momentum for NCE
PRINT_FREQ=10                  # Print frequency
RESUME_CHECKPOINT=""           # Path to checkpoint file if resuming

# Additional flags (adjust as needed)
EVALUATE=""                    # Add `--evaluate` if you want to evaluate only
SYNTHESIS_FLAG=""              # Add `--synthesis` to enable synthesis
MULTIAUG_FLAG="--multitask --multiaug"  # Add `--multiaug` to enable multi-augmentation
DOMAIN_FLAG=""                 # Add `--domain` to enable domain-based multitasking

# Define a function to sample log-uniform values
sample_loguniform() {
    low=$1
    high=$2
    size=$3
    coefficient=$4
    base=$5
    
    # Avoid generating zero or negative values by adjusting bounds
    min_value=1e-6  # Small positive value to avoid zero
    
    # Generate a log-uniform random value between low and high
    scale=$(python3 -c "import numpy as np; print(np.random.uniform($low, $high))")
    
    # Ensure scale is always greater than the minimum value (min_value)
    scale=$(echo "$scale" | awk -v min_value=$min_value '{if ($1 < min_value) print min_value; else print $1}')
    
    # Apply the coefficient and base to scale the result
    result=$(python3 -c "print($scale * $coefficient * $base)")
    
    # Ensure the result is positive and greater than zero
    result=$(echo "$result" | awk -v min_value=$min_value '{if ($1 < min_value) print min_value; else print $1}')
    
    # Output the result
    echo $result
}

# Loop over random lr and wd values
for SEED in $(seq $SEED_START $SEED_END); do
    # Sample learning rate and weight decay randomly in the desired range
    LR=$(sample_loguniform -5 -2 1 3 10)
    WD=$(sample_loguniform -6 -3 1 4 10)

    echo "Running with LR=$LR and WD=$WD for SEED=$SEED"

    # Construct the Python command with the current lr and wd
    python3 kaggle_main.py \
        "$DATA_PATH" \
        --arch "$ARCH" \
        --workers "$WORKERS" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --weight-decay "$WD" \
        --print-freq "$PRINT_FREQ" \
        --resume "$RESUME_CHECKPOINT" \
        --low-dim "$LOW_DIM" \
        --nce-k "$NCE_K" \
        --nce-t "$NCE_T" \
        --nce-m "$NCE_M" \
        --result "$RESULT_DIR" \
        --seed "$SEED" \
        $EVALUATE \
        $SYNTHESIS_FLAG \
        $MULTIAUG_FLAG \
        $DOMAIN_FLAG
done
