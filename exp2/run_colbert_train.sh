#!/bin/bash

# Get the number of clusters from the first argument, default to 4000
NUM_CLUSTERS=${1:-2000}

# Validate the cluster size
if [[ "$NUM_CLUSTERS" != "1000" && "$NUM_CLUSTERS" != "2000" && "$NUM_CLUSTERS" != "4000" && "$NUM_CLUSTERS" != "8000" ]]; then
    echo "Error: Unsupported number of clusters '$NUM_CLUSTERS'."
    echo "Usage: $0 [1000|2000|4000|8000]"
    exit 1
fi

echo "Starting Clustered ColBERT training with $NUM_CLUSTERS clusters..."

# Data paths
TRAIN_DATA="/home/slimelab/Projects/neural_lexical/data/training_data_v1_final_dedup.jsonl"
EVAL_DATA="/home/slimelab/Projects/neural_lexical/data/amazon_triplets.jsonl"

# Model configuration
BASE_MODEL="answerdotai/ModernBERT-large"
MODEL_TYPE="modernbert-large"
OUTPUT_DIR="./outputs_clustered_colbert_modernbert_large_${NUM_CLUSTERS}_relu_scale20_reg5e-3"
RUN_NAME="clustered-colbert-modernbert-large-${NUM_CLUSTERS}-relu_scale20_reg5e-3"

# Training Hyperparameters
BATCH_SIZE=64
MINI_BATCH_SIZE=16
LR=5e-5
EPOCHS=4
GRAD_ACC=1
NUM_WORKERS=8
OPTIM="adamw_torch"

# Launch training via the training script
torchrun --nproc_per_node=4 train_clustered_colbert.py \
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
    --scale 20 \
    --reg_weight 5e-3