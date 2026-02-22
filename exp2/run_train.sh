#!/bin/bash

# Define paths
TRAIN_DATA="/home/slime-base/projects/jian/neural_lexical/data/training_data_v1_final_dedup.jsonl"
EVAL_DATA="/home/slime-base/projects/jian/neural_lexical/data/amazon_triplets.jsonl"
BASE_MODEL="answerdotai/ModernBERT-large"
MODEL_TYPE="modernbert-large"
OUTPUT_DIR="./outputs_clustered_modernbert_large"
RUN_NAME="clustered-splade-modernbert-large"

# Training Hyperparameters
EPOCHS=2
BATCH_SIZE=256
GRAD_ACC=1
NUM_WORKERS=8
LR=5e-5
MINI_BATCH_SIZE=128

# Launch training via the training script
python3 -u train_clustered_splade.py \
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
    --scale 1.0 \
    --use_wandb \
    --use_triton
