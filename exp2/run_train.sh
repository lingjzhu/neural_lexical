#!/bin/bash

# Configuration: Set MODEL to "modernbert", "qwen3", or "t5gemma2"
MODEL=${1:-"llada"}

# Define paths
TRAIN_DATA="/home/slimelab/Projects/neural_lexical/data/training_data_v1_final_dedup.jsonl"
EVAL_DATA="/home/slimelab/Projects/neural_lexical/data/amazon_triplets.jsonl"

if [ "$MODEL" = "modernbert" ]; then
    BASE_MODEL="answerdotai/ModernBERT-large"
    MODEL_TYPE="modernbert-large"
    OUTPUT_DIR="./outputs_clustered_modernbert_large_4000_relu_reg5e-3"
    RUN_NAME="clustered-splade-modernbert-large-4000-relu-reg5e-3"
    BATCH_SIZE=256
    MINI_BATCH_SIZE=128
    LR=5e-5
    NUM_CLUSTERS=4000
    REG_WEIGHT=5e-4
    OPTIM="adamw_torch"
elif [ "$MODEL" = "qwen3" ]; then
    BASE_MODEL="Qwen/Qwen3-0.6B" 
    MODEL_TYPE="qwen3"
    OUTPUT_DIR="./outputs_clustered_qwen3_0.6B_4000_relu"
    RUN_NAME="clustered-splade-qwen3-0.6B-4000-relu"
    BATCH_SIZE=256 # Adjusted for likely memory constraints
    MINI_BATCH_SIZE=128
    LR=1e-4 # Qwen often benefits from slightly higher LR
    NUM_CLUSTERS=4000
    REG_WEIGHT=0
    OPTIM="adamw_torch"
elif [ "$MODEL" = "qwen3_diffusion" ]; then
    BASE_MODEL="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
    MODEL_TYPE="qwen3_diffusion"
    OUTPUT_DIR="./outputs_clustered_qwen3_diffusion_0.6B_4000_relu"
    RUN_NAME="clustered-splade-qwen3-diffusion-0.6B-4000-relu"
    BATCH_SIZE=256
    MINI_BATCH_SIZE=128
    LR=1e-4
    NUM_CLUSTERS=4000
    REG_WEIGHT=0
    OPTIM="adamw_torch"
elif [ "$MODEL" = "llada" ]; then
    BASE_MODEL="GSAI-ML/LLaDA-8B-Base"
    MODEL_TYPE="llada"
    OUTPUT_DIR="./outputs_clustered_llada_8B_4000_relu"
    RUN_NAME="clustered-splade-llada-8B-4000-relu-bf16"
    BATCH_SIZE=32          # significantly reduced for pure BF16
    MINI_BATCH_SIZE=8
    LR=1e-5
    NUM_CLUSTERS=4000
    REG_WEIGHT=0
    OPTIM="adamw_torch"
elif [ "$MODEL" = "t5gemma2" ]; then
    BASE_MODEL="google/t5gemma-2-1b-1b"
    MODEL_TYPE="t5gemma2"
    OUTPUT_DIR="./outputs_clustered_t5gemma2_1b_4000_relu"
    RUN_NAME="clustered-splade-t5gemma2-1b-4000-relu"
    BATCH_SIZE=16
    MINI_BATCH_SIZE=2
    LR=5e-5
    NUM_CLUSTERS=4000
    REG_WEIGHT=0
    OPTIM="adamw_bnb_8bit"
elif [ "$MODEL" = "qwen3_1.7b" ]; then
    BASE_MODEL="Qwen/Qwen3-1.7B"
    MODEL_TYPE="qwen3"
    OUTPUT_DIR="./outputs_clustered_qwen3_1.7B_4000_relu"
    RUN_NAME="clustered-splade-qwen3-1.7B-4000-relu"
    BATCH_SIZE=256
    MINI_BATCH_SIZE=64
    LR=5e-5
    NUM_CLUSTERS=4000
    REG_WEIGHT=0
    OPTIM="adamw_torch"
else
    echo "Unknown MODEL: $MODEL"
    exit 1
fi

# Training Hyperparameters
EPOCHS=2
GRAD_ACC=1
NUM_WORKERS=8

# Launch training via the training script
torchrun --nproc_per_node=4 train_clustered_splade.py \
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
    --use_triton \
    --num_clusters $NUM_CLUSTERS \
    --optim "$OPTIM" \
    --activation relu \
    --scale_embeddings False \
    --cluster_update_method greedy \
    --update_freq 10 \
    --unfreeze_embeddings \
    --reg_weight $REG_WEIGHT
