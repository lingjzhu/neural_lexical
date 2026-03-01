# Goal
 - Add `llada` as a base model for sentence transformer finetuning, with efficiency improvements from unsloth.

 # Instruction
 - Work inside `exp2` directory.
 - You can use the `modeling_llada.py` file from `/home/slime-base/projects/jian/learn/dllm/dllm/pipelines/llada/models/modeling_llada.py`  as a reference for how to implement the model.
 - You can use the `/home/slime-base/projects/jian/learn/unsloth/unsloth/models/sentence_transformer.py` as a reference for how to implement the unsloth efficiency improvements.
 - You need to enable fp8 patching and finetuning as is in `/home/slime-base/projects/jian/learn/unsloth` repo. Also enable kernel patching from unsloth. You should also enable lora finetuning with base model in fp8 mode. The unsloth repo should already have most of them available. But you need to add llada support. 
 - After implementing the model, you need to add it to the `train_clustered_splade.py` script as a base model option, so can it can be finetuned with the same script.
 - produce a benchmarking script to compare the fp8 model with the bf16 model of `GSAI-ML/LLaDA-8B-Base` and lora mode in forward and backward pass, in terms of precision, speed and memory usage. You might get OOM. So you can use a small batch size and sequence length. 