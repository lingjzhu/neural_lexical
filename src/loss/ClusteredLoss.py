"""Symmetric contrastive loss used for lexicon model training."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather
from sentence_transformers.util import dot_score


class ClusteredRankingLoss(nn.Module):
    def __init__(self, model, scale=1.0, similarity_fct=dot_score):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct

    def compute_loss_from_embeddings(self, embeddings, labels=None):
        if embeddings[0].dim() == 3:
            # A common sequence length permits concatenation and distributed gathering.
            length = self.model[0].max_seq_length
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
