#!/bin/bash

# Get the number of clusters from the first argument, default to 2000
NUM_CLUSTERS=${1:-2000}

# Validate the cluster size
if [[ "$NUM_CLUSTERS" != "1000" && "$NUM_CLUSTERS" != "2000" && "$NUM_CLUSTERS" != "4000" && "$NUM_CLUSTERS" != "8000" ]]; then
    echo "Error: Unsupported number of clusters '$NUM_CLUSTERS'."
    echo "Usage: $0 [1000|2000|4000|8000]"
    exit 1
fi

echo "Starting Clustered ColBERT distillation with $NUM_CLUSTERS clusters..."

# Data paths (using the distillation dataset you've generated)
TRAIN_DATA="/home/slimelab/Projects/neural_lexical/data/distillation_dataset_teacher.jsonl"
EVAL_DATA="/home/slimelab/Projects/neural_lexical/data/distillation_dataset_teacher.jsonl"

# Model configuration
# Note: Set BASE_MODEL to your pre-trained clustered colbert checkpoint if you have one, 
# for example: BASE_MODEL="./outputs_clustered_colbert_modernbert_large_2000_relu/checkpoint-1580"
BASE_MODEL="/home/slimelab/Projects/neural_lexical/exp2/outputs_clustered_colbert_modernbert_large_2000_relu/checkpoint-1580"
MODEL_TYPE="modernbert-large"
OUTPUT_DIR="./outputs_distill_modernbert_large_${NUM_CLUSTERS}_relu_reg5e-3_0303"
RUN_NAME="distill-colbert-modernbert-large-${NUM_CLUSTERS}-relu-reg5e-3-0303"

# Distillation parameters
MSE_WEIGHT=1.0
MNRL_WEIGHT=1.0
TEACHER_SCORE_SCALE=1.0

# Training Hyperparameters
BATCH_SIZE=64
MINI_BATCH_SIZE=8
LR=2e-5
EPOCHS=2
GRAD_ACC=1
NUM_WORKERS=8
OPTIM="adamw_torch"

# Launch training via the distillation script
torchrun --nproc_per_node=4 distill_clustered_colbert.py \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --base_model "$BASE_MODEL" \
    --model_type "$MODEL_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --mini_batch_size $MINI_BATCH_SIZE \
    --grad_acc $GRAD_ACC \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --use_wandb \
    --use_triton \
    --num_clusters $NUM_CLUSTERS \
    --optim "$OPTIM" \
    --activation relu \
    --scale_embeddings False \
    --cluster_update_method greedy \
    --update_freq 10 \
    --unfreeze_embeddings \
    --scale 1 \
    --reg_weight 5e-3 \
    --mse_weight $MSE_WEIGHT \
    --mnrl_weight $MNRL_WEIGHT \
    --teacher_score_scale $TEACHER_SCORE_SCALE
