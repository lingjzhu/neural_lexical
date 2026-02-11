from __future__ import annotations

from enum import Enum
from typing import Callable

from numpy import ndarray
from torch import Tensor

from .scores import (
    colbert_scores,
    colbert_scores_pairwise,
    colbert_scores_masked_mean,
    colbert_scores_pairwise_masked_mean,
)


class SimilarityFunction(Enum):
    """
    Enum class for supported score functions.

    Supported values:
    - ``MAXSIM``: ColBERT max-sim (IR-style, asymmetric)
    - ``MASKED_MEAN``: Symmetric masked pairwise mean (AV-style)
    """

    MAXSIM = "MaxSim"
    MASKED_MEAN = "MaskedMean"

    @staticmethod
    def to_similarity_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[..., Tensor]:
        """
        Converts a similarity function name or enum value to the corresponding similarity function.
        """
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.MAXSIM:
            return colbert_scores

        if similarity_function == SimilarityFunction.MASKED_MEAN:
            return colbert_scores_masked_mean

        raise ValueError(
            f"The provided function {similarity_function} is not supported. "
            f"Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def to_similarity_pairwise_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[..., Tensor]:
        """
        Converts a similarity function into a pairwise similarity function.
        """
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.MAXSIM:
            return colbert_scores_pairwise

        if similarity_function == SimilarityFunction.MASKED_MEAN:
            return colbert_scores_pairwise_masked_mean

        raise ValueError(
            f"The provided function {similarity_function} is not supported. "
            f"Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def possible_values() -> list[str]:
        """
        Returns a list of possible values for the SimilarityFunction enum.
        """
        return [m.value for m in SimilarityFunction]
