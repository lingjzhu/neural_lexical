# Implementation Plan: Engram-Style Sparse Representation Head

## 0. Tips
 - Keep everything inside the `/home/slime-base/projects/jian/neural_lexical/experimental` directory.
 - You have access to the `engram.py` implementation of engram. You can use it directly. Most of the descriptions below are from this implementation. 
 - You have access to the `modeling_modernbert.py` implementation of modernbert.
 - Use modernbert as the base model. But make necessary modifications to support the implementation. If the existing module can be used, like FFN and attention and engram layer, don't rewrite them.
 - Use engram as the style representation head.
 - Add your model in a new `.py` file.
 - Add tests to test forward and backward passes. You can initialize a small-scale model for local testing. Then give me a tutorial in jupyter notebook.
 - Keep the code clean and modular for human readability

## 1. Objective

To implement a neural output head that transforms Transformer hidden states into a sparse, interpretable set of ** weighted N-gram embeddings**. This representation will be used for **Authorship Verification (AV)** via a late-interaction MaxSim operator.

Here is a structured implementation plan for building the hybrid **Dual-MemoryLLM** (combining MemoryLLM's initial token processing with Engram's gated n-gram lookups).

### **Phase 1: Data Preparation & N-Gram Indexing**

Before modifying the transformer layers, the input pipeline must be updated to handle multi-token n-gram hashing alongside standard tokenization.

* **Tokenizer Extension:** Modify the standard tokenizer to output not just the token IDs (), but also sliding windows of n-grams (e.g., 2-grams and 3-grams) for every token position.
* **Hashing Mechanism:** Implement a fast, deterministic hashing function (like MurmurHash3) to map every generated n-gram string to a specific integer index within a massive, predefined vocabulary size for the n-gram memory table.
* **Memory Table Initialization:** Initialize a large embedding table on the host CPU RAM to store the n-gram vectors. This will act as the external knowledge base.

### **Phase 2: Architectural Modification (The Forward Pass)**

Redesign the transformer block to support the three distinct parallel pathways: Contextual Reasoning, Phrase-Level Memory, and Token-Level Memory.

* **Pathway 1: Contextual Reasoning (Self-Attention)**
* Input: The evolving residual stream ().
* Operation: Standard Multi-Head Self-Attention.
* Output: Context-aware hidden state ().


* **Pathway 2: Phrase-Level Memory (Gated N-Gram Lookup)**
* Input: Hashed n-gram indices for the current sequence.
* Operation: Fetch the n-gram embeddings from the external table. Use a lightweight linear layer with a Sigmoid activation on  to generate a scalar gate. Multiply the fetched embeddings by this gate.
* Output: Contextually relevant phrase-facts ().


* **Pathway 3: Token-Level Memory (MemoryLLM FFN)**
* Input: The initial, static token embeddings from the very first layer ().
* Operation: Pass  through a standard Feed-Forward Network.
* Output: Unconditional word-level facts ().


* **Fusion Step:** Sum the outputs of all three pathways to create the input for the next layer.


For the final output, you can use the following code as reference, which also relies on engram.
```
import torch
import torch.nn as nn
import math

class AuthorshipEngramHead(nn.Module):
    def __init__(self, hidden_size, engram_hidden_size, k_selection=128):
        super().__init__()
        self.k = k_selection
        self.hidden_size = hidden_size
        
        # We assume hash_mapping and multi_head_embedding are handled prior 
        # or instantiated here similar to the official code.
        
        # DeepSeek's Q-K Gating Projections
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_proj = nn.Linear(engram_hidden_size, hidden_size)
        
        # RMSNorms for stable dot-products
        self.norm_key = nn.RMSNorm(hidden_size)
        self.norm_query = nn.RMSNorm(hidden_size)

    def forward(self, hidden_states, engram_embeddings):
        """
        hidden_states: [B, L, D] - The dynamic contextual sequence
        engram_embeddings: [B, L, Engram_D] - The static fetched N-grams
        """
        B, L, _ = hidden_states.shape
        
        # 1. Prepare Keys (Static) and Queries (Dynamic)
        key = self.key_proj(engram_embeddings)
        normed_key = self.norm_key(key)
        
        normed_query = self.norm_query(hidden_states)
        
        # 2. Compute Official DeepSeek Gate Scores
        # Dot product scaled by sqrt of hidden dimension
        raw_gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
        
        # The signed square-root stabilization trick
        stable_gate = raw_gate.abs().clamp_min(1e-6).sqrt() * raw_gate.sign()
        
        # Final gate evaluated between 0 and 1
        # Shape: [B, L]
        gate_scores = stable_gate.sigmoid() 
        
        # 3. Apply Gating (The Value Projection)
        # Shape: [B, L, D]
        gated_values = gate_scores.unsqueeze(-1) * self.value_proj(engram_embeddings)
        
        # --- AUTHORSHIP SPECIFIC LOGIC BEGINS HERE ---
        # Notice we completely omit the self.short_conv(value)
        
        # 4. Global Top-K Selection across the sequence length L
        # We use the gate_scores as the relevance metric for the stylistic fingerprint
        topk_scores, topk_indices = torch.topk(gate_scores, k=self.k, dim=1)
        
        # Expand indices to gather the full D-dimensional embeddings
        # Shape: [B, K, D]
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        # Extract the final representation for late interaction
        final_authorship_representation = torch.gather(gated_values, dim=1, index=gather_indices)
        
        return final_authorship_representation, topk_indices
```
