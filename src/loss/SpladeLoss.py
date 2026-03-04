from __future__ import annotations

import logging
from collections.abc import Iterable
import math

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import FlopsLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

logger = logging.getLogger(__name__)

from sentence_transformers import util

class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return (x,)
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return grads[0]
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

def gather_tensor(tensor):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensors = GatherLayer.apply(tensor)
        
        if tensors[0].dim() == 3: # Handle sequence tensors [B, S, C]
            max_len = max([t.shape[1] for t in tensors])
            padded = []
            for t in tensors:
                if t.shape[1] < max_len:
                    pad_shape = (t.shape[0], max_len - t.shape[1], t.shape[2])
                    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                    padded.append(torch.cat([t, pad], dim=1))
                else:
                    padded.append(t)
            tensors = padded
            
        return torch.cat(tensors, dim=0)
    return tensor


class SparseSelfMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model, scale: float = 1.0, similarity_fct=util.dot_score):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor = None) -> torch.Tensor:
        raise AttributeError("Use compute_loss_from_embeddings directly.")

    def compute_loss_from_embeddings(self, embeddings: list[torch.Tensor], labels: torch.Tensor = None) -> torch.Tensor:
        anchors = gather_tensor(embeddings[0])
        candidates_list = [gather_tensor(e) for e in embeddings[1:]]
        batch_size = anchors.size(0)
        num_candidates = len(candidates_list)

        candidates = torch.cat(candidates_list, dim=0)
        
        # Pad anchors and candidates to the same sequence length if they are 3D
        if anchors.dim() == 3 and candidates.dim() == 3:
            max_len = max(anchors.shape[1], candidates.shape[1])
            if anchors.shape[1] < max_len:
                pad_shape = (anchors.shape[0], max_len - anchors.shape[1], anchors.shape[2])
                pad = torch.zeros(pad_shape, dtype=anchors.dtype, device=anchors.device)
                anchors = torch.cat([anchors, pad], dim=1)
            if candidates.shape[1] < max_len:
                pad_shape = (candidates.shape[0], max_len - candidates.shape[1], candidates.shape[2])
                pad = torch.zeros(pad_shape, dtype=candidates.dtype, device=candidates.device)
                candidates = torch.cat([candidates, pad], dim=1)

        # 1. Q loss: Q tries to find D_pos among [D_pos, D_neg, ..., Q]
        q_candidates = torch.cat([candidates, anchors], dim=0)
        q_scores = self.similarity_fct(anchors, q_candidates) * self.scale
        
        idx = torch.arange(batch_size, device=anchors.device)
        # Mask out Q[i] @ Q[i] which is at offset num_candidates * batch_size
        q_scores[idx, num_candidates * batch_size + idx] = float('-inf')
        loss_q = self.cross_entropy_loss(q_scores, idx)

        # 2. D loss: D_pos tries to find Q among [Q, D_pos, D_neg, ...]
        positives = candidates_list[0]
        d_candidates = torch.cat([anchors, candidates], dim=0)
        d_scores = self.similarity_fct(positives, d_candidates) * self.scale
        
        # Mask out D_pos[i] @ D_pos[i] which is at offset batch_size
        d_scores[idx, batch_size + idx] = float('-inf')
        loss_d = self.cross_entropy_loss(d_scores, idx)

        return (loss_q + loss_d) / 2


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
        return w


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
        
        self._reg_step += 1

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

from contextlib import nullcontext
from functools import partial
import tqdm
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import RandContext

def _cached_splade_backward_hook(
    grad_output: torch.Tensor,
    sentence_features: Iterable[dict[str, torch.Tensor]],
    loss_obj: "CachedSpladeMixedTopKLoss",
) -> None:
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for seq_idx, (sentence_feature, grad, random_states) in enumerate(zip(sentence_features, loss_obj.cache, loss_obj.random_states)):
            bsz = sentence_feature["input_ids"].size(0)
            for (reps_dict, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                reps_mb_sparse = reps_dict["sparse_embeddings"]
                if reps_mb_sparse.requires_grad:
                    surrogate = (reps_mb_sparse.flatten() * grad_mb.flatten()).sum() * grad_output

                    mbsz = reps_mb_sparse.size(0)
                    current_reg_w = loss_obj._get_regularizer_weight()

                    if loss_obj.use_document_regularizer_only or seq_idx > 0:
                        reg_loss = loss_obj.document_regularizer.compute_loss_from_embeddings(reps_mb_sparse)
                        surrogate = surrogate + reg_loss * current_reg_w * (mbsz / bsz) * grad_output
                    elif seq_idx == 0 and loss_obj.query_regularizer_weight is not None:
                        reg_loss = loss_obj.query_regularizer.compute_loss_from_embeddings(reps_mb_sparse)
                        surrogate = surrogate + reg_loss * current_reg_w * (mbsz / bsz) * grad_output

                    surrogate.backward()
            
            # Optional: Clear cache to prevent fragmentation in large-scale GradCache training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class CachedSpladeMixedTopKLoss(SpladeMixedTopKLoss):
    def __init__(self, mini_batch_size: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mini_batch_size = mini_batch_size
        self.cache = None
        self.random_states = None
        self.show_progress_bar = False

    def embed_minibatch(
        self,
        sentence_feature: dict[str, torch.Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext = None,
    ):
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, torch.Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] = None,
    ):
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.shape[0]
        for i, b in enumerate(range(0, bsz, self.mini_batch_size)):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[torch.Tensor]]) -> torch.Tensor:
        loss = self.calculate_contrastive_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def calculate_contrastive_loss(self, reps: list[list[torch.Tensor]], with_backward: bool = False) -> torch.Tensor:
        anchors = gather_tensor(torch.cat(reps[0], dim=0))
        candidates_list = [gather_tensor(torch.cat(r, dim=0)) for r in reps[1:]] 
        candidates = torch.cat(candidates_list, dim=0)
        
        batch_size = anchors.size(0)
        num_candidates = len(candidates_list)
        idx = torch.arange(batch_size, device=anchors.device)

        q_candidates = torch.cat([candidates, anchors], dim=0)
        q_scores = self.sparse_loss.similarity_fct(anchors, q_candidates) * self.sparse_loss.scale
        q_scores[idx, num_candidates * batch_size + idx] = float('-inf')
        loss_q = self.sparse_loss.cross_entropy_loss(q_scores, idx)

        positives = candidates_list[0]
        d_candidates = torch.cat([anchors, candidates], dim=0)
        d_scores = self.sparse_loss.similarity_fct(positives, d_candidates) * self.sparse_loss.scale
        d_scores[idx, batch_size + idx] = float('-inf')
        loss_d = self.sparse_loss.cross_entropy_loss(d_scores, idx)

        total_loss = (loss_q + loss_d) / 2

        if with_backward:
            total_loss.backward()
            total_loss = total_loss.detach()
        return total_loss
        
    def calculate_reg_loss_no_grad(self, reps, seq_idx):
        corpus_reps = torch.cat(reps[seq_idx], dim=0)
        if seq_idx > 0 or self.use_document_regularizer_only:
            return self.document_regularizer.compute_loss_from_embeddings(corpus_reps).detach()
        else:
            return self.query_regularizer.compute_loss_from_embeddings(corpus_reps).detach()

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        reps = []
        self.random_states = []
        
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb_dict, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True
            ):
                sparse_emb = reps_mb_dict["sparse_embeddings"].detach().requires_grad_()
                reps_mbs.append(sparse_emb)
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        losses = {}
        current_reg_w = self._get_regularizer_weight()

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(partial(_cached_splade_backward_hook, sentence_features=sentence_features, loss_obj=self))
            losses["sparse_loss"] = loss
            
            # Detached logic for logging
            if self.use_document_regularizer_only:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(len(reps))) / len(reps)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
            else:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(1, len(reps))) / max(1, len(reps)-1)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
                if self.query_regularizer_weight is not None:
                    query_loss = self.calculate_reg_loss_no_grad(reps, 0)
                    losses["query_regularizer_loss"] = query_loss * current_reg_w
        else:
            losses["sparse_loss"] = self.calculate_contrastive_loss(reps)
            if self.use_document_regularizer_only:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(len(reps))) / len(reps)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
            else:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(1, len(reps))) / max(1, len(reps)-1)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
                if self.query_regularizer_weight is not None:
                    query_loss = self.calculate_reg_loss_no_grad(reps, 0)
                    losses["query_regularizer_loss"] = query_loss * current_reg_w

        if self.sparse_loss is not None:
            self.sparse_loss.scale = self._get_scale()
            self._step += 1
        
        self._reg_step += 1

        return losses
