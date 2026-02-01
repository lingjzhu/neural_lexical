from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import FlopsLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from src.kernels.fused_maxsim import sparse_maxsim


logger = logging.getLogger(__name__)


class ColbertMultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        scale: float = 1.0,
        similarity_fct=None,
        gather_across_devices: bool = False,
    ):
        super().__init__()

        if similarity_fct is None:
            raise ValueError("You must provide a similarity function for sparse embeddings.")

        self.model = model
        self.scale = scale
        self._similarity_fct = similarity_fct

    def compute_loss_from_embeddings(self, embeddings, labels=None, masks=None):
        """
        embeddings: list of embeddings from encode()
                    where each embedding is (vals, inds)
        """

        anchors = embeddings[0]
        positives = embeddings[1]

        batch_size = anchors[0].size(0)

        if masks is not None:
            anchor_mask = masks[0]
            positive_mask = masks[1]


        sims = self._similarity_fct(anchors, positives, anchor_mask, positive_mask)
        
        sims = sims * self.scale


        labels = torch.arange(batch_size, device=sims.device)
        loss = torch.nn.functional.cross_entropy(sims, labels)

        return loss

    # Keep the original restriction for raw forward()
    def forward(self, *args, **kwargs):
        raise AttributeError("Use inside SpladeLoss instead of directly calling forward().")


class SpladeColbertTopKLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        sparse_loss: nn.Module = None,
        scale: float = 1.0,
        document_regularizer_weight: float | None = 0,
        query_regularizer_weight: float | None = None,
        document_regularizer: nn.Module | None = None,
        query_regularizer: nn.Module | None = None,
        document_regularizer_threshold: int | None = None,
        query_regularizer_threshold: int | None = None,
        use_document_regularizer_only: bool = False,
        similarity_fn_name: str = "MaxSim"
    ):
 
        super().__init__()
        self.model = model
        if similarity_fn_name == "MaxSim":
            self.similarity_fct = sparse_maxsim
        elif similarity_fn_name == "MaskedMean":
            # NOTE: sparse_masked_mean was not defined in the original combined file either.
            # Keeping the logic for consistency.
            try:
                from src.kernels.fused_maxsim import sparse_masked_mean
                self.similarity_fct = sparse_masked_mean
            except ImportError:
                logging.error("sparse_masked_mean not found in .fused_maxsim")
                raise
        self.sparse_loss = ColbertMultipleNegativesRankingLoss(model,similarity_fct=self.similarity_fct, scale=scale)
        self.document_regularizer_weight = document_regularizer_weight
        self.query_regularizer_weight = query_regularizer_weight
        self.use_document_regularizer_only = use_document_regularizer_only


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

        

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:

        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature) for sentence_feature in sentence_features]
        sparse_embeddings = [embedding["sparse_embeddings"] for embedding in embeddings]
        masks = [embedding["attention_mask"] for embedding in embeddings]

        losses = {}
        base_loss = 0.0


        base_loss = self.sparse_loss.compute_loss_from_embeddings(sparse_embeddings, None, masks)

        losses['sparse_loss'] = base_loss


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
