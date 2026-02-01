from __future__ import annotations

import logging
from collections.abc import Iterable
import math

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import FlopsLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

logger = logging.getLogger(__name__)


class SpladeMixedTopKLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        dense_loss: nn.Module,
        sparse_loss: nn.Module,
        k: list | None = [100],
        dense_weight: float = 2.0,
        aux_weight: float = 0.0,
        document_regularizer_weight: float | None = 0,
        query_regularizer_weight: float | None = None,
        document_regularizer: nn.Module | None = None,
        query_regularizer: nn.Module | None = None,
        document_regularizer_threshold: int | None = 256,
        query_regularizer_threshold: int | None = 256,
        use_document_regularizer_only: bool = False,
        logit_soft_capping: float = 0.0,
        scale_start: float = 1.0,
        scale_end: float = 1.0,
        total_steps: int = 10000,
        reg_start: float = 0.0,
        reg_total_steps: int = 10000,
    ):
        """
        SpladeLoss implements the loss function for the SPLADE (Sparse Lexical and Expansion) model,
        which combines a main loss function with regularization terms to control efficiency.

        This loss function balances effectiveness (via the main loss) with efficiency by regularizing
        both the query and document representations to be sparse, reducing computational requirements
        at inference time.

        Args:
            model: SparseEncoder model
            loss: The principal loss function to use can be any of the SparseEncoder losses except CSR related losses and flops loss.
            document_regularizer_weight: Weight for the corpus regularization term. This term encourages sparsity in the document embeddings.
                Will be applied to positive documents and all negatives one if some are provided. In some papers, this parameter is
                referred to as "lambda_d" (document) or "lambda_c" (corpus).
            query_regularizer_weight: Weight for the query regularization term. This term encourages sparsity in the query embeddings.
                If None, no query regularization will be applied, it's not a problem if you are in an inference-free setup or
                if you are having use_document_regularizer_only=True. Else you should have a query_regularizer_weight > 0.
                In some papers, this parameter is referred to as "lambda_q" (query).
            document_regularizer: Optional regularizer to use specifically for corpus regularization instead of the default FlopsLoss.
                This allows for different regularization strategies for documents vs queries.
            query_regularizer: Optional regularizer to use specifically for query regularization instead of the default FlopsLoss.
                This allows for different regularization strategies for queries vs documents.
            document_regularizer_threshold: Optional threshold for the number of non-zero (active) elements in the corpus embeddings to be considered in the FlopsLoss.
                If specified, only corpus embeddings with more than this number of non-zero (active) elements will be considered.
                Only used when document_regularizer is None (for the default FlopsLoss).
            query_regularizer_threshold: Optional threshold for the number of non-zero (active) elements in the query embeddings to be considered in the FlopsLoss.
                If specified, only query embeddings with more than this number of non-zero (active) elements will be considered.
                Only used when query_regularizer is None (for the default FlopsLoss).
            use_document_regularizer_only: If True, all input embeddings are treated as documents and regularized together with document_regularizer_weight.
                Especially useful when training with symmetric texts (e.g. pairs of documents) or more.

        References:
            - For more details, see the paper "From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective"
              https://arxiv.org/abs/2205.04733

        Requirements:
            1. Input requirements depend on the chosen loss
            2. Usually used with a teacher model in a knowledge distillation setup and an associated loss

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("distilbert/distilbert-base-uncased")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "passage1": ["It's so sunny.", "He took the car to work."],
                        "passage2": ["It's very sunny.", "She walked to the store."],
                    }
                )

                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_passages1 = teacher_model.encode(batch["passage1"])
                    emb_passages2 = teacher_model.encode(batch["passage2"])
                    return {
                        "label": teacher_model.similarity_pairwise(emb_queries, emb_passages1)
                        - teacher_model.similarity_pairwise(emb_queries, emb_passages2)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SpladeLoss(
                    student_model,
                    loss=losses.SparseMarginMSELoss(student_model),
                    document_regularizer_weight=3e-5,
                    query_regularizer_weight=5e-5,
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.dense_loss = dense_loss
        self.sparse_loss = sparse_loss
        self.document_regularizer_weight = document_regularizer_weight
        self.query_regularizer_weight = query_regularizer_weight
        self.use_document_regularizer_only = use_document_regularizer_only
        self.dense_weight = dense_weight
        self.k = k
        self.aux_weight = aux_weight
        self.logit_soft_capping_value = logit_soft_capping

        # scale scheduling
        self.scale_start = scale_start
        self.scale_end   = scale_end
        self.total_steps = max(1, total_steps)
        self._step = 0

        self.reg_start = reg_start
        self.reg_end = document_regularizer_weight
        self.reg_total_steps = max(1, reg_total_steps)

        # internal counter
        self._reg_step = 0

        # Set up regularizers with defaults to FlopsLoss using specific thresholds
        self.document_regularizer = (
            document_regularizer
            if document_regularizer is not None
            else FlopsLoss(model, threshold=document_regularizer_threshold)
        )
        if query_regularizer is not None:
            self.query_regularizer = query_regularizer
        elif not use_document_regularizer_only:
            self.query_regularizer = FlopsLoss(model, threshold=query_regularizer_threshold)

        if self.query_regularizer_weight is None and not use_document_regularizer_only:
            logging.warning(
                "query_regularizer_weight is None. This means that the query regularization will not be applied. If you are in an inference free set up it's fine else you should have a query_regularizer_weight > 0."
            )
        if self.use_document_regularizer_only and self.query_regularizer_weight is not None:
            logging.warning(
                "query_regularizer_weight should be None when use_document_regularizer_only is True. use_document_regularizer_only mean we consider all the input to be of the same type and so under the same regularization. query_regularizer_weight will be ignored."
            )
            self.query_regularizer_weight = None

        
    def topk_sparse(self, x, k):
        # x: (batch_size, dim)
        # returns a tensor of same shape, with only top-k values per row kept
        values, indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x).scatter_(1, indices, 1.0)
        return x * mask

    def _get_scale(self):

        s = min(1.0, self._step / self.total_steps)

        scale = self.scale_start + s * (self.scale_end - self.scale_start)

        return max(1.0, min(self.scale_end, scale))
    
    def _get_regularizer_weight(self):

        s = min(1.0, self._reg_step / self.reg_total_steps)
        w = self.reg_start + s * (self.reg_end - self.reg_start)

        return min(self.scale_end, w)


    def logit_soft_capping(self,logits,temp=1.0):
        logits = logits / temp
        logits = torch.tanh(logits)
        logits = logits * temp
        return logits
    
    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature) for sentence_feature in sentence_features]
        sparse_embeddings = [embedding["sparse_embeddings"] for embedding in embeddings]
        dense_embeddings = [embedding["dense_embeddings"] for embedding in embeddings]


        losses = {}
        base_loss = 0.0
        aux_loss = 0.0

        current_reg_w = self._get_regularizer_weight()

        if self.use_document_regularizer_only:
            # If use_document_regularizer_only is True, we consider all the input to be of the same type and so under the same regularization
            corpus_loss = self.document_regularizer.compute_loss_from_embeddings(torch.cat(sparse_embeddings))
        else:
            corpus_loss = self.document_regularizer.compute_loss_from_embeddings(torch.cat(sparse_embeddings[1:]))
        losses["document_regularizer_loss"] = corpus_loss * current_reg_w

        # Add query regularization if enabled
        if self.query_regularizer_weight is not None:
            query_loss = self.query_regularizer.compute_loss_from_embeddings(sparse_embeddings[0])
            losses["query_regularizer_loss"] = query_loss * current_reg_w

        if self.sparse_loss is not None:
            if self.logit_soft_capping_value > 0.0:
                sparse_embeddings = [self.logit_soft_capping(embedding, temp=self.logit_soft_capping_value) for embedding in sparse_embeddings]
            if self.k is None:
                base_loss += self.sparse_loss.compute_loss_from_embeddings(sparse_embeddings, labels)
            else:
                for k_i in self.k:
                    k_embeddings = [self.topk_sparse(embedding, k_i) for embedding in sparse_embeddings]
                    base_loss += self.sparse_loss.compute_loss_from_embeddings(k_embeddings, labels)/len(self.k)
                    if self.aux_weight > 0.0:
                        residue_embeddings = [self.topk_sparse(sparse_embeddings[l] - k_embeddings[l], 1024)  for l in range(len(sparse_embeddings))]
                        aux_loss += self.aux_weight*self.sparse_loss.compute_loss_from_embeddings(residue_embeddings, labels)/len(self.k)

            losses['sparse_loss'] = base_loss
            if self.aux_weight > 0.0:
                losses['aux_loss'] = aux_loss

        if self.dense_loss is not None:
            dense_loss = self.dense_weight*self.dense_loss.compute_loss_from_embeddings(dense_embeddings, labels)
            losses['dense_loss'] = dense_loss
        
        # update scales
        if self.sparse_loss is not None:
            scale = self._get_scale()
            self.sparse_loss.scale = scale
            self._step += 1

        return losses

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        config_dict = {
            "loss": self.sparse_loss,
            "document_regularizer_weight": self.document_regularizer_weight,
        }
        if self.query_regularizer_weight is not None:
            config_dict["query_regularizer_weight"] = self.query_regularizer_weight
        # Include regularizer names (if not flops) and threshold information (if not None)

        if not isinstance(self.document_regularizer, FlopsLoss):
            config_dict["document_regularizer"] = self.document_regularizer.__class__.__name__
        if hasattr(self.document_regularizer, "threshold") and self.document_regularizer.threshold is not None:
            config_dict["document_regularizer_threshold"] = self.document_regularizer.threshold

        if hasattr(self, "query_regularizer") and self.query_regularizer is not None:
            if not isinstance(self.query_regularizer, FlopsLoss):
                config_dict["query_regularizer"] = self.query_regularizer.__class__.__name__
            if hasattr(self.query_regularizer, "threshold") and self.query_regularizer.threshold is not None:
                config_dict["query_regularizer_threshold"] = self.query_regularizer.threshold
        return config_dict
