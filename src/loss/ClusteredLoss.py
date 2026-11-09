"""Symmetric contrastive loss used for lexicon model training."""
from contextlib import nullcontext
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather
from sentence_transformers.util import dot_score
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import RandContext


class ClusteredRankingLoss(nn.Module):
    def __init__(self, model, scale=1.0, similarity_fct=dot_score):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct

    def compute_loss_from_embeddings(self, embeddings, labels=None):
        if embeddings[0].dim() == 3:
            # A common sequence length permits concatenation and distributed gathering.
            length = getattr(self.model, "module", self.model)[0].max_seq_length
            embeddings = [F.pad(e, (0, 0, 0, length - e.shape[1])) for e in embeddings]
        if torch.distributed.is_initialized():
            embeddings = [torch.cat(all_gather(e)) for e in embeddings]
        anchors, positives = embeddings[:2]
        candidates = torch.cat(embeddings[1:])
        count = len(anchors)
        indices = torch.arange(count, device=anchors.device)
        q_scores = self.similarity_fct(anchors, torch.cat([candidates, anchors])) * self.scale
        d_scores = self.similarity_fct(positives, torch.cat([anchors, candidates])) * self.scale
        q_scores[indices, len(candidates) + indices] = -torch.inf
        d_scores[indices, count + indices] = -torch.inf
        return (F.cross_entropy(q_scores, indices) + F.cross_entropy(d_scores, indices)) / 2


class ClusteredTrainingLoss(nn.Module):
    """Symmetric ranking plus FLOPS regularization, with optional warmup."""
    def __init__(self, model, scale=1.0, reg_weight=0.005, reg_warmup_steps=200):
        super().__init__()
        from src.models.ClusteredTransformer import clustered_maxsim
        similarity = clustered_maxsim if model[0].embedding_type == "colbert" else dot_score
        self.ranking = ClusteredRankingLoss(model, scale, similarity)
        self.model = model
        self.reg_weight = reg_weight
        self.reg_warmup_steps = reg_warmup_steps
        self.step = 0

    def loss_from_embeddings(self, embeddings):
        loss = self.ranking.compute_loss_from_embeddings(embeddings)
        if self.reg_weight:
            weight = self.reg_weight * min(1.0, self.step / max(1, self.reg_warmup_steps))
            # For token vectors, regularize the cluster dimensions over all tokens.
            flops = [e.float().reshape(-1, e.shape[-1]).abs().mean(0).square().sum()
                     for e in embeddings]
            loss = loss + weight * (flops[0] + torch.stack(flops[1:]).mean())
        return loss

    def forward(self, sentence_features, labels=None):
        return self.loss_from_embeddings([self.model(dict(f))["sparse_embeddings"]
                                          for f in sentence_features])

    def get_config_dict(self):
        return {"scale": self.ranking.scale, "reg_weight": self.reg_weight,
                "reg_warmup_steps": self.reg_warmup_steps}


# Reuse Sentence Transformers' minibatch cache and RNG replay machinery.
class CachedClusteredTrainingLoss(CachedMultipleNegativesRankingLoss):
    def __init__(self, model, mini_batch_size=32, **kwargs):
        super().__init__(model, mini_batch_size=mini_batch_size)
        self.objective = ClusteredTrainingLoss(model, **kwargs)
        self.autocast_context = ("cpu", False, torch.bfloat16)

    def embed_minibatch(self, sentence_feature, begin, end, with_grad,
                        copy_random_state, random_state=None):
        features = {k: v[begin:end] if isinstance(v, torch.Tensor) else v
                    for k, v in sentence_feature.items()}
        device = features["input_ids"].device.type
        if copy_random_state:
            self.autocast_context = (device, torch.is_autocast_enabled(device),
                                     torch.get_autocast_dtype(device))
        device, enabled, dtype = self.autocast_context
        with (random_state if random_state is not None else nullcontext()):
            with (torch.enable_grad() if with_grad else torch.no_grad()):
                state = RandContext(*features.values()) if copy_random_state else None
                with torch.autocast(device, enabled=enabled, dtype=dtype):
                    reps = self.model(features)["sparse_embeddings"]
        return reps, state

    def forward(self, sentence_features, labels=None):
        features = list(sentence_features)
        reps, self.random_states = [], []
        for feature in features:
            batches = list(self.embed_minibatch_iter(feature, with_grad=False, copy_random_state=True))
            reps.append([r.detach().requires_grad_() for r, _ in batches])
            self.random_states.append([state for _, state in batches])
        if not torch.is_grad_enabled():
            return self.calculate_loss(reps)
        loss = self.calculate_loss_and_cache_gradients(reps)
        loss.register_hook(partial(self.replay_gradients, features=features))
        return loss

    def replay_gradients(self, grad_output, features):
        with torch.enable_grad():
            for feature, gradients, states in zip(features, self.cache, self.random_states):
                batches = self.embed_minibatch_iter(feature, with_grad=True,
                                                    copy_random_state=False, random_states=states)
                for (reps, _), gradient in zip(batches, gradients):
                    # Elementwise reduction also supports autocast token tensors on CUDA.
                    surrogate = (reps.float() * gradient.float()).sum() * grad_output
                    surrogate.backward()

    def calculate_loss(self, reps, with_backward=False):
        loss = self.objective.loss_from_embeddings([torch.cat(r) for r in reps])
        if with_backward:
            loss.backward()
            loss = loss.detach()
        return loss

    def get_config_dict(self):
        return {**self.objective.get_config_dict(), "mini_batch_size": self.mini_batch_size}
