#!/bin/bash

EXP=$1
CKPT="checkpoint-1582"

if [ -z "$EXP" ]; then
    echo "Usage: ./run_eval.sh <exp_id>"
    echo "Example: ./run_eval.sh 1"
    exit 1
fi

# Configuration based on Exp ID
if [ "$EXP" == "1" ]; then
    NAME="clustered_modernbert_large_4000"
    NUM_CLUSTERS=4000
    ACTIVATION="log1p_relu"
    BASE_MODEL="answerdotai/ModernBERT-large"
    MODEL_TYPE="modernbert"
elif [ "$EXP" == "2" ]; then
    NAME="clustered_modernbert_large_8000_fixed"
    NUM_CLUSTERS=8000
    ACTIVATION="log1prelu"
    BASE_MODEL="answerdotai/ModernBERT-large"
    MODEL_TYPE="modernbert"
elif [ "$EXP" == "3" ]; then
    NAME="clustered_modernbert_large_4000_relu_reg5e-3"
    NUM_CLUSTERS=4000
    ACTIVATION="relu"
    BASE_MODEL="answerdotai/ModernBERT-large"
    MODEL_TYPE="modernbert"
elif [ "$EXP" == "4" ]; then
    NAME="clustered_qwen3_0.6B_4000_relu"
    NUM_CLUSTERS=4000
    ACTIVATION="relu"
    BASE_MODEL="Qwen/Qwen3-0.6B"
    MODEL_TYPE="qwen3"
elif [ "$EXP" == "5" ]; then
    NAME="clustered_qwen3_diffusion_0.6B_4000_relu"
    NUM_CLUSTERS=4000
    ACTIVATION="relu"
    BASE_MODEL="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
    MODEL_TYPE="qwen3_diffusion"
else
    echo "Unknown Experiment ID: $EXP"
    exit 1
fi

MODEL="./outputs_${NAME}/$CKPT"

echo "🚀 Evaluating $NAME ($CKPT) with Activation=$ACTIVATION..."

mkdir -p hrs_eval

python3 evaluate_hrs.py \
    --parent_dir "/home/slime-base/projects/jian/neural_lexical/data/hrs_release_11-22-24" \
    --model_name "$MODEL" \
    --base_model "$BASE_MODEL" \
    --num_clusters $NUM_CLUSTERS \
    --activation "$ACTIVATION" \
    --output_file "hrs_eval/${NAME}.json" \
    --clustered \
    --use_triton \
    --model_type "$MODEL_TYPE" \
    --k $NUM_CLUSTERS \
    --disable_preds
