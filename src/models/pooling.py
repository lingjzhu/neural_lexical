from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor
from sentence_transformers.models.Module import Module

logger = logging.getLogger(__name__)


class SpladePooling(Module):
    """
    SPLADE Pooling module for creating the sparse embeddings.

    This module implements the SPLADE pooling mechanism that:

    1. Takes token logits from a masked language model (MLM).
    2. Applies a sparse transformation using an activation function followed by log1p (i.e., log(1 + activation(MLM_logits))).
    3. Applies a pooling strategy `max` or `sum` to produce sparse embeddings.

    The resulting embeddings are highly sparse and capture lexical information,
    making them suitable for efficient information retrieval.

    Args:
        pooling_strategy (str): Pooling method across token dimensions.
            Choices:
                - `sum`: Sum pooling (used in original SPLADE see https://arxiv.org/pdf/2107.05720).
                - `max`: Max pooling (used in SPLADEv2 and later models see https://arxiv.org/pdf/2109.10086 or https://arxiv.org/pdf/2205.04733).
        activation_function (str): Activation function applied before log1p transformation.
            Choices:
                - `relu`: ReLU activation (standard in all Splade models).
                - `log1p_relu`: log(1 + ReLU(x)) variant used in Opensearch Splade models, see https://arxiv.org/pdf/2504.14839.
        word_embedding_dimension (int, optional): Dimensionality of the output embeddings (if needed).
        chunk_size (int, optional): Chunk size along the sequence length dimension (i.e., number of tokens per chunk).
            If None, processes entire sequence at once. Using smaller chunks the reduces memory usage but may
            lower the training and inference speed. Default is None.
    """

    SPLADE_POOLING_MODES = ("sum", "max", "mean")
    SPLADE_ACTIVATION = ["relu", "log1p_relu"]
    config_keys: list[str] = ["pooling_strategy", "activation_function", "word_embedding_dimension"]

    def __init__(
        self,
        pooling_strategy: Literal["max", "sum", "mean"] = "max",
        activation_function: Literal["relu", "log1p_relu"] = "relu",
        word_embedding_dimension: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.activation_function = activation_function
        if activation_function not in self.SPLADE_ACTIVATION:
            raise ValueError("activation_function must be either 'relu' or 'log1p_relu'")
        self.word_embedding_dimension = word_embedding_dimension  # This will be set in the forward method
        self.chunk_size = chunk_size

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            features: Dictionary containing input features. Expects:
                - 'token_embeddings': MLM logits (shape: batch_size, seq_length, vocab_size).
                - 'attention_mask': Attention mask (shape: batch_size, seq_length).
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        mlm_logits = features["token_embeddings"]
        attention_mask = features["attention_mask"]  # Shape: [batch_size, seq_length]

        # Unsqueeze attention_mask to be [batch_size, seq_length, 1] for broadcasting
        attention_mask_expanded = attention_mask.unsqueeze(-1).to(mlm_logits.dtype)

        batch_size, seq_len, vocab_s = mlm_logits.shape
        device = mlm_logits.device

        # Initialize pooled scores based on pooling strategy
        if self.pooling_strategy == "max":
            pooled_scores = torch.full((batch_size, vocab_s), float("-inf"), dtype=mlm_logits.dtype, device=device)
        elif self.pooling_strategy == "sum" or self.pooling_strategy == "mean":
            pooled_scores = torch.zeros((batch_size, vocab_s), dtype=mlm_logits.dtype, device=device)
        else:
            raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

        # Process in chunks if chunk_size is set, otherwise process the entire sequence at once
        chunk_size = seq_len if (self.chunk_size is None or self.chunk_size <= 0) else self.chunk_size

        for i in range(0, seq_len, chunk_size):
            try:
                current_chunk_logits = mlm_logits[:, i : i + chunk_size, :]
                current_chunk_mask = attention_mask_expanded[:, i : i + chunk_size, :]

                masked_current_chunk_logits = current_chunk_logits * current_chunk_mask

                current_chunk_transformed = masked_current_chunk_logits.relu_()
                if not self.training:
                    current_chunk_transformed = current_chunk_transformed.log1p_()
                else:
                    current_chunk_transformed = current_chunk_transformed.log1p()
                # With "log1p_relu", we apply a second log1p
                if self.activation_function == "log1p_relu":
                    if not self.training:
                        current_chunk_transformed = current_chunk_transformed.log1p_()
                    else:
                        current_chunk_transformed = current_chunk_transformed.log1p()

                if self.pooling_strategy == "max":
                    chunk_pooled = torch.max(current_chunk_transformed, dim=1)[0]
                    pooled_scores = torch.maximum(pooled_scores, chunk_pooled)
                else:
                    chunk_pooled = torch.sum(current_chunk_transformed, dim=1)
                    pooled_scores += chunk_pooled
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "Ran out of memory during SpladePooling. "
                        "Consider setting or decreasing the 'chunk_size' parameter. "
                        "Smaller chunk_size reduces memory usage at the cost of slower processing, "
                        "but will allow for larger batch sizes."
                    )
                raise e
        
        if self.pooling_strategy == "mean":
            token_counts = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled_scores = pooled_scores / token_counts


        if self.word_embedding_dimension is None:
            self.word_embedding_dimension = pooled_scores.shape[1]
        features["sentence_embedding"] = pooled_scores
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    def __repr__(self) -> str:
        return f"SpladePooling({self.get_config_dict()})"

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embedding.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.word_embedding_dimension
    



class LightSpladePooling(Module):
    """
    SPLADE Pooling module for creating the sparse embeddings.

    This module implements the SPLADE pooling mechanism that:

    1. Takes token logits from a masked language model (MLM).
    2. Applies a sparse transformation using an activation function followed by log1p (i.e., log(1 + activation(MLM_logits))).
    3. Applies a pooling strategy `max` or `sum` to produce sparse embeddings.

    The resulting embeddings are highly sparse and capture lexical information,
    making them suitable for efficient information retrieval.

    Args:
        pooling_strategy (str): Pooling method across token dimensions.
            Choices:
                - `sum`: Sum pooling (used in original SPLADE see https://arxiv.org/pdf/2107.05720).
                - `max`: Max pooling (used in SPLADEv2 and later models see https://arxiv.org/pdf/2109.10086 or https://arxiv.org/pdf/2205.04733).
        activation_function (str): Activation function applied before log1p transformation.
            Choices:
                - `relu`: ReLU activation (standard in all Splade models).
                - `log1p_relu`: log(1 + ReLU(x)) variant used in Opensearch Splade models, see https://arxiv.org/pdf/2504.14839.
        word_embedding_dimension (int, optional): Dimensionality of the output embeddings (if needed).
        chunk_size (int, optional): Chunk size along the sequence length dimension (i.e., number of tokens per chunk).
            If None, processes entire sequence at once. Using smaller chunks the reduces memory usage but may
            lower the training and inference speed. Default is None.
    """

    SPLADE_POOLING_MODES = ("mean")
    SPLADE_ACTIVATION = ["relu", "log1p_relu"]
    config_keys: list[str] = ["pooling_strategy", "activation_function", "word_embedding_dimension"]

    def __init__(
        self,
        pooling_strategy: Literal["mean", "max"] = "mean",
        activation_function: Literal["relu", "log1p_relu"] = "relu",
        word_embedding_dimension: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.activation_function = activation_function
        if activation_function not in self.SPLADE_ACTIVATION:
            raise ValueError("activation_function must be either 'relu' or 'log1p_relu'")
        self.word_embedding_dimension = word_embedding_dimension  # This will be set in the forward method
        self.chunk_size = chunk_size

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            features: Dictionary containing input features. Expects:
                - 'token_embeddings': MLM logits (shape: batch_size, seq_length, vocab_size).
                - 'attention_mask': Attention mask (shape: batch_size, seq_length).
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        mlm_logits = features["sparse_embeddings"]
        attention_mask = features["attention_mask"]  # Shape: [batch_size, seq_length]

        batch_size, vocab_s = mlm_logits.shape
        device = mlm_logits.device

        try:
            if self.pooling_strategy != "mean":
                mlm_logits = mlm_logits.relu_()
            if not self.training:
                mlm_logits = mlm_logits.log1p_()
            else:
                mlm_logits = mlm_logits.log1p()
            # With "log1p_relu", we apply a second log1p
            if self.activation_function == "log1p_relu":
                if not self.training:
                    mlm_logits = mlm_logits.log1p_()
                else:
                    mlm_logits = mlm_logits.log1p()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "Ran out of memory during SpladePooling. "
                    "Consider setting or decreasing the 'chunk_size' parameter. "
                    "Smaller chunk_size reduces memory usage at the cost of slower processing, "
                    "but will allow for larger batch sizes."
                )
            raise e


        if self.word_embedding_dimension is None:
            self.word_embedding_dimension = mlm_logits.shape[1]
        features["sparse_embeddings"] = mlm_logits
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    def __repr__(self) -> str:
        return f"SpladePooling({self.get_config_dict()})"

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embedding.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.word_embedding_dimension
    


class SparseColbertPooling(Module):
    """
    SPLADE Pooling module for creating the sparse embeddings.

    This module implements the SPLADE pooling mechanism that:

    1. Takes token logits from a masked language model (MLM).
    2. Applies a sparse transformation using an activation function followed by log1p (i.e., log(1 + activation(MLM_logits))).
    3. Applies a pooling strategy `max` or `sum` to produce sparse embeddings.

    The resulting embeddings are highly sparse and capture lexical information,
    making them suitable for efficient information retrieval.

    Args:
        pooling_strategy (str): Pooling method across token dimensions.
            Choices:
                - `sum`: Sum pooling (used in original SPLADE see https://arxiv.org/pdf/2107.05720).
                - `max`: Max pooling (used in SPLADEv2 and later models see https://arxiv.org/pdf/2109.10086 or https://arxiv.org/pdf/2205.04733).
        activation_function (str): Activation function applied before log1p transformation.
            Choices:
                - `relu`: ReLU activation (standard in all Splade models).
                - `log1p_relu`: log(1 + ReLU(x)) variant used in Opensearch Splade models, see https://arxiv.org/pdf/2504.14839.
        word_embedding_dimension (int, optional): Dimensionality of the output embeddings (if needed).
        chunk_size (int, optional): Chunk size along the sequence length dimension (i.e., number of tokens per chunk).
            If None, processes entire sequence at once. Using smaller chunks the reduces memory usage but may
            lower the training and inference speed. Default is None.
    """

    SPLADE_POOLING_MODES = ("mean")
    SPLADE_ACTIVATION = ["relu", "log1p_relu"]
    config_keys: list[str] = ["pooling_strategy", "activation_function", "word_embedding_dimension"]

    def __init__(
        self,
        pooling_strategy: Literal["mean"] = "mean",
        activation_function: Literal["relu", "log1p_relu"] = "relu",
        word_embedding_dimension: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.activation_function = activation_function
        if activation_function not in self.SPLADE_ACTIVATION:
            raise ValueError("activation_function must be either 'relu' or 'log1p_relu'")
        self.word_embedding_dimension = word_embedding_dimension  # This will be set in the forward method
        self.chunk_size = chunk_size

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            features: Dictionary containing input features. Expects:
                - 'token_embeddings': MLM logits (shape: batch_size, seq_length, vocab_size).
                - 'attention_mask': Attention mask (shape: batch_size, seq_length).
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        mlm_logits, mlm_inds = features["sparse_embeddings"]
        attention_mask = features["attention_mask"]  # Shape: [batch_size, seq_length]

        #batch_size, vocab_s = mlm_logits.shape
        device = mlm_logits.device

        try:
            mlm_logits = torch.relu(mlm_logits)
            mlm_logits = mlm_logits.log1p()
            # With "log1p_relu", we apply a second log1p
            if self.activation_function == "log1p_relu":

                mlm_logits = mlm_logits.log1p()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "Ran out of memory during SpladePooling. "
                    "Consider setting or decreasing the 'chunk_size' parameter. "
                    "Smaller chunk_size reduces memory usage at the cost of slower processing, "
                    "but will allow for larger batch sizes."
                )
            raise e


        if self.word_embedding_dimension is None:
            self.word_embedding_dimension = mlm_logits.shape[1]
        features["sparse_embeddings"] = (mlm_logits, mlm_inds)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    def __repr__(self) -> str:
        return f"SpladePooling({self.get_config_dict()})"

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embedding.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.word_embedding_dimension
    



class Pooling(Module):
    """
    Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows
    to use the CLS token if it is returned by the underlying word embedding model. You can concatenate multiple poolings
    together.

    Args:
        word_embedding_dimension: Dimensions for the word embeddings
        pooling_mode: Either "cls", "lasttoken", "max", "mean",
            "mean_sqrt_len_tokens", or "weightedmean". If set,
            overwrites the other pooling_mode_* settings
        pooling_mode_cls_token: Use the first token (CLS token) as text
            representations
        pooling_mode_max_tokens: Use max in each dimension over all
            tokens.
        pooling_mode_mean_tokens: Perform mean-pooling
        pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but
            divide by sqrt(input_length).
        pooling_mode_weightedmean_tokens: Perform (position) weighted
            mean pooling. See `SGPT: GPT Sentence Embeddings for
            Semantic Search <https://huggingface.co/papers/2202.08904>`_.
        pooling_mode_lasttoken: Perform last token pooling. See `SGPT:
            GPT Sentence Embeddings for Semantic Search
            <https://huggingface.co/papers/2202.08904>`_ and `Text and Code
            Embeddings by Contrastive Pre-Training
            <https://huggingface.co/papers/2201.10005>`_.
        include_prompt: If set to false, the prompt tokens are not
            included in the pooling. This is useful for reproducing
            work that does not include the prompt tokens in the pooling
            like INSTRUCTOR, but otherwise not recommended.
    """

    POOLING_MODES = (
        "cls",
        "lasttoken",
        "max",
        "mean",
        "mean_sqrt_len_tokens",
        "weightedmean",
    )

    config_keys = [
        "word_embedding_dimension",
        "pooling_mode_cls_token",
        "pooling_mode_mean_tokens",
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
        "pooling_mode_lasttoken",
        "include_prompt",
    ]

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str | None = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
        include_prompt: bool = True,
    ) -> None:
        super().__init__()

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()

            if pooling_mode not in self.POOLING_MODES:
                raise ValueError(
                    f"Set invalid pooling mode: {pooling_mode}. Valid pooling modes are: {self.POOLING_MODES}."
                )

            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_mean_sqrt_len_tokens = pooling_mode == "mean_sqrt_len_tokens"
            pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
            pooling_mode_lasttoken = pooling_mode == "lasttoken"

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken

        self.include_prompt = include_prompt

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_weightedmean_tokens,
                pooling_mode_lasttoken,
            ]
        )
        self.pooling_output_dimension = pooling_mode_multiplier * word_embedding_dimension

    def __repr__(self) -> str:
        return f"Pooling({self.get_config_dict()})"

    def get_pooling_mode_str(self) -> str:
        """Returns the pooling mode as string"""
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt_len_tokens")
        if self.pooling_mode_weightedmean_tokens:
            modes.append("weightedmean")
        if self.pooling_mode_lasttoken:
            modes.append("lasttoken")

        return "+".join(modes)

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        token_embeddings = features["dense_embeddings"]
        attention_mask = (
            features["attention_mask"]
            if "attention_mask" in features
            else torch.ones(token_embeddings.shape[:-1], device=token_embeddings.device, dtype=torch.int64)
        )
        if not self.include_prompt and "prompt_length" in features:
            prompt_length = features["prompt_length"]
            # prompt_length is either:
            # * an int (in inference)
            # * a tensor of shape (bs), all the same value (in training with an IterableDataset)
            # * a tensor of shape (1) (in training with a Dataset)
            # We turn all into an int
            if isinstance(prompt_length, torch.Tensor):
                prompt_length = prompt_length[0].item()

            attention_mask[:, :prompt_length] = 0

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # Use flip and max() to get the last index of 1 in the attention mask

            if torch.jit.is_tracing():
                # Avoid tracing the argmax with int64 input that can not be handled by ONNX Runtime: https://github.com/microsoft/onnxruntime/issues/10068
                attention_mask = attention_mask.to(torch.int32)

            values, indices = attention_mask.flip(1).max(1)
            indices = torch.where(values == 0, seq_len - 1, indices)
            gather_indices = seq_len - indices - 1

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features["dense_embeddings"] = output_vector
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.pooling_output_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)