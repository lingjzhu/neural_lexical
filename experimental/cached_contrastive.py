from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tqdm
from torch import Tensor
from torch.utils.checkpoint import get_device_states, set_device_states

# --- From pylate.utils.distributed ---


not_init_warning = True
logger = logging.getLogger(__name__)

_has_warned_dist_not_initialized = False


def all_gather(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """Gathers a tensor from each distributed rank into a list. The tensor for the local rank is the original one, with the gradients while the others have no gradients.

    - If torch.distributed is available and initialized:
      1. Creates a list of tensors (each sized like the input `tensor`).
      2. Gathers tensors from each rank into that list.
      3. Replaces the local tensor in the list with the original tensor that retains gradients.

    - If torch.distributed is either unavailable, uninitialized, or
      `world_size == 1`, it returns a list containing only the
      original tensor and throws a warning to notify the user (helpful when using a single GPU setup).

    Parameters
    ----------
    tensor:
        The input tensor to be gathered from each rank.

    Returns
    -------
    Sequence:
        A list of tensors collected from each rank. On a single GPU or when distributed is uninitialized, the list will contain only the original tensor.

    """
    global _has_warned_dist_not_initialized

    # Check if torch.distributed is properly available and initialized.
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

        # Perform all_gather.
        dist.all_gather(gathered_tensors, tensor)

        # Replace local rank's tensor with the original (retaining gradients).
        local_rank = dist.get_rank()
        gathered_tensors[local_rank] = tensor
        return gathered_tensors

    # Warn once about uninitialized or single-GPU usage.
    if not _has_warned_dist_not_initialized:
        warning = """
            Trying to gather while torch.distributed is not available or has not been initialized,
             returning the original (local) tensor. This is expected if you are
             only using one GPU; consider not using gathering to remove this warning.
       """
        logger.warning(warning)
        _has_warned_dist_not_initialized = True

    return [tensor]


def all_gather_with_gradients(tensor: torch.Tensor) -> Sequence[torch.Tensor]:
    """Gathers a tensor from each distributed rank into a list. All the tensors will retain gradients.
    This is the same as `all_gather`, but all the tensors will retain gradients and is used to compute contrastive with local queries only to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616

    - If torch.distributed is available and initialized, gather all the tensors (with gradients) from each rank into a list

    - If torch.distributed is either unavailable, uninitialized, or
      `world_size == 1`, it returns a list containing only the
      original tensor and throws a warning to notify the user (helpful when using a single GPU setup).

    Parameters
    ----------
    tensor:
        The input tensor to be gathered from each rank.

    Returns
    -------
    Sequence:
        A list of tensors collected from each rank. On a single GPU or when distributed is uninitialized, the list will contain only the original tensor.

    """
    global _has_warned_dist_not_initialized

    # Check if torch.distributed is properly available and initialized.
    if dist.is_available() and dist.is_initialized():
        tensor = dist.nn.all_gather(tensor)
        return tensor

    # Warn once about uninitialized or single-GPU usage.
    if not _has_warned_dist_not_initialized:
        warning = """
            Trying to gather while torch.distributed is not available or has not been initialized,
             returning the original (local) tensor. This is expected if you are
             only using one GPU; consider not using gathering to remove this warning.
       """
        logger.warning(warning)
        _has_warned_dist_not_initialized = True

    return [tensor]


def get_rank() -> int:
    """Returns the current rank in a distributed training."""
    # Check if torch.distributed is properly available and initialized.
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Returns the world size in a distributed training."""
    # Check if torch.distributed is properly available and initialized.
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# --- From pylate.utils.tensor ---

def convert_to_tensor(
    x: torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray | list | float],
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    if isinstance(x, list):
        if not x:
            return torch.tensor([], dtype=torch.float32)

        if isinstance(x[0], np.ndarray):
            return torch.from_numpy(np.array(x, dtype=np.float32))

        if isinstance(x[0], list):
            return torch.tensor(x, dtype=torch.float32)

        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
    return torch.tensor(x)

# --- From pylate.scores.scores ---
def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the ColBERT scores between queries and documents embeddings. The score is computed as the sum of maximum similarities
    between the query and the document.

    Parameters
    ----------
    queries_embeddings
        The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
    documents_embeddings
        The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    queries_mask
        The mask for the queries embeddings. Shape: (batch_size, num tokens queries)
    documents_mask
        The mask for the documents embeddings. Shape: (batch_size, num tokens documents)

    Examples
    --------
    >>> import torch

    >>> queries_embeddings = torch.tensor([
    ...     [[1.], [0.], [0.], [0.]],
    ...     [[0.], [2.], [0.], [0.]],
    ...     [[0.], [0.], [3.], [0.]],
    ... ])

    >>> documents_embeddings = torch.tensor([
    ...     [[10.], [0.], [1.]],
    ...     [[0.], [100.], [10.]],
    ...     [[1.], [0.], [1000.]],
    ... ])

    >>> documents_mask = torch.tensor([
    ...     [1., 1., 1.],
    ...     [1., 0., 1.],
    ...     [1., 1., 1.],
    ... ])
    >>> query_mask = torch.tensor([
    ...     [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 0., 1.]
    ... ])

    >>> scores = colbert_scores(
    ...     queries_embeddings=queries_embeddings,
    ...     documents_embeddings=documents_embeddings,
    ...     queries_mask=query_mask,
    ...     documents_mask=documents_mask,
    ... )

    >>> scores
    tensor([[  10.,  10., 1000.],
            [  20.,  20., 2000.],
            [  0.,  0., 0.]])

    """
    queries_embeddings = convert_to_tensor(queries_embeddings)
    documents_embeddings = convert_to_tensor(documents_embeddings)

    scores = torch.einsum(
        "ash,bth->abst",
        queries_embeddings,
        documents_embeddings,
    )

    if queries_mask is not None:
        queries_mask = convert_to_tensor(queries_mask)
        scores = scores * queries_mask.unsqueeze(1).unsqueeze(3)
        

    if documents_mask is not None:
        documents_mask = convert_to_tensor(documents_mask)
        scores = scores * documents_mask.unsqueeze(0).unsqueeze(2)
    scores = scores.max(axis=-1).values.sum(axis=-1)

    if queries_mask is not None:
        qlen = queries_mask.sum(dim=1).clamp_min(1)  # (B,)
        scores = scores / qlen.unsqueeze(1)
    return scores

# --- From pylate.losses.cached_contrastive ---
class RandContext:
    """Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        if torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend is not supported for this operation. Please use CPU or CUDA."
            )
        else:
            self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj,
) -> None:
    """A backward hook that re-runs the forward for each mini-batch with gradients enabled
    and uses the cached partial derivatives w.r.t. the embeddings to backprop.
    """
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(
            sentence_features, loss_obj.cache, loss_obj.random_states
        ):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                # Plug back the cached gradients into the backward pass by dotting them with the corresponding representations
                # Scale by grad_output from the top-level backward pass to account for gradient of downstream operations (not useful in this setup)
                surrogate = (
                    torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                )
                surrogate.backward()


class CachedContrastive(nn.Module):
    """A cached, in-batch negatives contrastive loss for PyLate, analogous to
    SentenceTransformers' CachedMultipleNegativesRankingLoss. This allows
    large effective batch sizes by chunking the embeddings pass and caching
    gradients w.r.t. those embeddings.

    Parameters
    ----------
    model :
        A PyLate ColBERT model
    score_metric
        ColBERT scoring function. Defaults to colbert_scores.
    mini_batch_size
        Chunk size for the forward pass. You can keep this small to avoid OOM on large batch sizes.
    size_average
        Whether to average or sum the cross-entropy loss across the mini-batch.
    gather_across_devices
        Whether to gather the embeddings across devices to have more in batch negatives. We recommend making sure the sampling across GPUs use the same dataset in case of multi-dataset training to make sure the negatives are plausible.
    show_progress_bar
        Whether to show a TQDM progress bar for the embedding steps.

    Examples
    --------
    >>> from pylate import models, losses

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = losses.CachedContrastive(model=model, mini_batch_size=1)

    >>> anchors = model.tokenize([
    ...     "fruits are healthy.", "chips are not healthy."
    ... ], is_query=True)

    >>> positives = model.tokenize([
    ...     "fruits are good for health.", "chips are not good for health."
    ... ], is_query=False)

    >>> negatives = model.tokenize([
    ...     "fruits are bad for health.", "chips are good for health."
    ... ], is_query=False)

    >>> sentence_features = [anchors, positives, negatives]

    >>> loss = loss(sentence_features=sentence_features)
    >>> assert isinstance(loss.item(), float)
    """

    def __init__(
        self,
        model: nn.Module,
        score_metric: Callable = colbert_scores,
        mini_batch_size: int = 32,
        size_average: bool = True,
        gather_across_devices: bool = False,
        show_progress_bar: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super(CachedContrastive, self).__init__()
        self.model = model
        self.score_metric = score_metric
        self.mini_batch_size = mini_batch_size
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.show_progress_bar = show_progress_bar
        self.temperature = temperature

        # Will hold partial derivatives for each embedding chunk
        self.cache: list[list[Tensor]] | None = None
        # Will hold random states for each chunk, so we can re-run the embedding pass with grads
        self.random_states: list[list[RandContext]] | None = None

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Forward pass on a slice [begin:end] of sentence_feature. If 'with_grad' is False,
        we run under torch.no_grad. If 'copy_random_state' is True, we create and return
        a RandContext so that we can exactly reproduce this forward pass later.
        """
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            k: v[begin:end] for k, v in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                # If we need a new random-state copy, create it
                random_state = (
                    RandContext(*sentence_feature_minibatch.values())
                    if copy_random_state
                    else None
                )
                outputs = self.model(sentence_feature_minibatch)
                # by default, PyLate ColBERT forward returns a dict with "token_embeddings"
                embeddings = F.normalize(outputs["token_embeddings"], p=2, dim=-1, eps=1e-8)

        return embeddings, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Yields chunks of embeddings (and corresponding RandContext) for the given
        sentence_feature, respecting the mini_batch_size limit.
        """
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps, masks) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        # we want partial grads on all the chunked embeddings
        loss = self.calculate_loss(reps, masks, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [
            [r.grad for r in rs] for rs in reps
        ]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps, masks, with_backward: bool = False) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients. Each sub-list in reps is a list of mini-batch chunk embeddings

        Parameters
        ----------
        reps :
            A list of list of mini-batch chunk embeddings. The first list are the anchors, the second are the positives and the remaining are negatives.
        masks
            Tensors containing the skiplist masks associated with each sentence feature (anchor, positives, negatives).
        with_backward
            Whether to compute the backward pass or not.
        """
        # We first cat them chunk-wise for anchor, positives, negatives
        embeddings_anchor = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_other = [
            torch.cat([chunk_embed for chunk_embed in r]) for r in reps[1:]
        ]  # [(nneg * bsz, hdim)]

        batch_size = len(embeddings_anchor)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=reps[0][0].device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        # Possibly gather the embeddings across devices to have more in-batch negatives. For GradCache, we only need to gather them to compute the scores matrix and nowhere else.
        # Note that we only gather the documents embeddings and not the queries embeddings, but are keeping gradients. This is to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616
        if self.gather_across_devices:
            embeddings_other = [
                torch.cat(all_gather_with_gradients(embeddings))
                for embeddings in embeddings_other
            ]
            # Masks [0] is the anchor mask so we do not need to gather it (even though we are not using it for now anyways)
            # Also, we do gather without gradients for the masks as we do not backpropagate through them
            masks = [
                masks[0],
                *[torch.cat(all_gather(mask)) for mask in masks[1:]],
            ]
            rank = get_rank()
            # Adjust the labels to match the gathered embeddings positions
            labels = labels + rank * batch_size
        losses: list[torch.Tensor] = []
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            end = begin + self.mini_batch_size
            # We chunk the scores computation to avoid OOM because MaxSim can get expensive with large batch sizes/long documents
            scores = torch.cat(
                [
                    torch.cat(
                        [
                            self.score_metric(
                                embeddings_anchor[begin:end],
                                group_embeddings[
                                    g_start : min(
                                        g_start + self.mini_batch_size,
                                        len(group_embeddings),
                                    )
                                ],
                                queries_mask=masks[0][begin:end],
                                documents_mask=documents_mask[
                                    g_start : min(
                                        g_start + self.mini_batch_size,
                                        len(group_embeddings),
                                    )
                                ],
                            )
                            for g_start in range(
                                0, len(group_embeddings), self.mini_batch_size
                            )
                        ],
                        dim=1,
                    )
                    for group_embeddings, documents_mask in zip(
                        embeddings_other, masks[1:]
                    )
                ],
                dim=1,
            )
            # We don't want to average the loss across the mini-batch as mini-batch sizes can vary, which would create an issue similar to this one: https://huggingface.co/blog/gradient_accumulation#where-does-it-stem-from
            loss_mbatch = F.cross_entropy(
                input=scores / self.temperature,
                target=labels[begin:end],
                reduction="sum",
            )
            # Scale by world size when gathering across device
            if self.gather_across_devices:
                loss_mbatch *= get_world_size()

            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        if self.size_average:
            loss /= batch_size

        return loss

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the CachedConstrastive loss.

        Parameters
        ----------
        sentence_features
            List of tokenized sentences. The first sentence is the anchor and the rest are the positive and negative examples.
        labels
            The labels for the contrastive loss. Not used in this implementation, but kept for compatibility with Trainer.
        """
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        # handle the model being wrapped in (D)DP and so require to access module first
        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = [sentence_feature["attention_mask"] for sentence_feature in sentence_features]

        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)
        # Update masks to match the actual embeddings length (e.g. K_selection = 128)
        updated_masks = []
        for r_mbs, orig_mask in zip(reps, masks):
            embed_len = r_mbs[0].shape[1]
            b_size = sum(r.shape[0] for r in r_mbs)
            
            if orig_mask.shape[1] != embed_len:
                # The model truncated representations (like 128 top-k facts)
                # Compute mask based on how many valid tokens were passed, up to embed_len
                valid_lens = orig_mask.sum(dim=1).clamp(max=embed_len).long()
                new_mask = torch.zeros((b_size, embed_len), dtype=orig_mask.dtype, device=orig_mask.device)
                for i, vlen in enumerate(valid_lens):
                    new_mask[i, :vlen] = 1
                updated_masks.append(new_mask)
            else:
                updated_masks.append(orig_mask)
        masks = updated_masks

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps, masks)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(
                partial(
                    _backward_hook, sentence_features=sentence_features, loss_obj=self
                )
            )
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps, masks)

        return loss

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""

