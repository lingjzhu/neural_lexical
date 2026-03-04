# Goal
 - Train colbert model using the clustered embedding approach. Replace the dense embeddings with the sparse clustered embeddings like in `neural_lexical/exp2/train_clustered_splade.py`.

 # Instruction
 - Work inside `neural_lexical/exp2` directory. You can continue to use the `llada-lora` conda environment.
 - You can refer to the `neural_lexical/src/train_sparse_colbert.py`  as a reference for how to implement the model. However, in the clustered approach, please use the full projected clustered embeddings, instead of the topk values in the original sparse colbert implementation. The cluster ids should also be updated like in clustered splade. Essentially you are simply extending the clustered splade into colbert. You can also consult `neural_lexical/pylate/pylate/scores/scores.py` for implementing the colbert scoring. 
 - You should enable colbert training with `modern-bert` as base model. No need to work on other models for now. Also enable tricks like flash attention or sdpa, and gradient checkpointing to save memory. 
 - After implementing the model, you need to add it to the `train_clustered_colbert.py` script for training. Ensure that the CacheContrastive loss can be used. Ensure that Q-Q and D-D negatives are available like in the `train_clustered_splade.py` script. Ensure that distributed training across 4 GPUs is working.
 - Please test your code to make sure that it is fully functonal. 