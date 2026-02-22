import torch
from modeling_dual_memoryllm import DualMemoryLLMConfig, DualMemoryLLMForAuthorshipVerification

# Use a tiny ModernBERT model or standard 
# Using a community tiny model to save download time or just the config if model exists
model_id = "answerdotai/ModernBERT-base" 

# First just test config load
try:
    config = DualMemoryLLMConfig.from_pretrained(model_id)
    # We add engram-specific args that might not be in the base config
    config.engram_layer_ids = [1, 3]
    config.engram_vocab_size = [1000, 1000]
    
    print("Loading weights...")
    model = DualMemoryLLMForAuthorshipVerification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
    print("Successfully loaded ModernBERT weights into DualMemoryLLM!")
except Exception as e:
    print(f"Failed: {e}")
