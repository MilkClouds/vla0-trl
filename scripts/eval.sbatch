#!/bin/bash
#SBATCH --job-name=eval-vla0
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load any necessary modules or activate environment here
# module load cuda/11.8
# . ~/miniforge3/bin/activate owa

cd ../
source .venv/bin/activate

# Config
MODEL_PATH="./runs/vla0_sft-lr4e-5-gradclip/checkpoint-200000"
LOG_DIR="./eval_logs/vla0_sft-lr4e-5-gradclip/checkpoint-200000/horizon-1"
NUM_SHARDS=50
NUM_TASKS=8
MAX_SUBTASKS=7  # ceil(NUM_SHARDS / NUM_TASKS)

# Round-robin distribution for balanced load:
srun -u bash -c '
for i in $(seq 0 $(('"$MAX_SUBTASKS"' - 1))); do
    SHARD_ID=$((SLURM_PROCID + i * '"$NUM_TASKS"'))

    # Skip if shard_id >= num_shards
    if [ $SHARD_ID -ge '"$NUM_SHARDS"' ]; then
        continue
    fi

    python scripts/eval.py \
        --model_path '"$MODEL_PATH"' \
        --log_dir '"$LOG_DIR"' \
        --action_horizon 1 \
        --ensemble_prediction 0 \
        --torch_compile \
        --skip_evaluated \
        --shard_id $SHARD_ID --num_shards '"$NUM_SHARDS"' &
done

wait
'