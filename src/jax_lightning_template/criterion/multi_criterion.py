"""
This file contains the implementation of the MultiCriterion class.
"""

from typing import Any, List

import equinox

from ..utils import utils
from ..utils.typing import CriterionConfig


class MultiCriterion(equinox.Module):
    """
    This class represents a multi-criterion.

    Args:
        criterions (List[CriterionConfig | str | equinox.nn.Module]): The criterions.
        criterion_weights (list | None): The weights of the criterions. Defaults to None.
        reduce (str): The reduction method. Defaults to "mean".

    Raises:
        ValueError: If the reduce is invalid.

    Examples:
        >>> criterions = [
        ...     {"criterion": "equinox.nn.CrossEntropyLoss", "criterion_params": {}},
        ...     "equinox.nn.MSELoss",
        ...     equinox.nn.L1Loss(),
        ... ]
        >>> criterion_weights = [1, 2]
        >>> reduce = "mean"
        >>> multi_criterion = EquinoxMultiCriterion(criterions, criterion_weights, reduce)
    """

    criterions: List[equinox.Module]
    criterion_weights: List[float]
    reduce: str

    def __init__(
        self,
        criterions: List[CriterionConfig | str | equinox.Module],
        criterion_weights: list | None = None,
        reduce: str = "mean",
    ):
        super().__init__()
        self.criterions = []
        for criterion in criterions:
            if isinstance(criterion, dict):
                self.criterions.append(
                    utils.get_instance(
                        criterion["criterion"], criterion["criterion_params"]
                    )
                )
            elif isinstance(criterion, str):
                self.criterions.append(utils.get_instance(criterion, {}))
            elif isinstance(criterion, equinox.Module):
                self.criterions.append(criterion)
        self.criterion_weights = (
            criterion_weights
            if criterion_weights is not None
            else [1] * len(self.criterions)
        )
        self.reduce = reduce
        # assert all modules callable
        assert all(
            callable(criterion) for criterion in self.criterions
        ), "All criterions must be callable."

    def __call__(self, x: Any, y: Any):
        """
        This function calculates the loss.

        Args:
            x (Any): The input.
            y (Any): The target.

        Returns:
            jax.numpy.ndarray: The loss.
        """

        losses = []
        for i, criterion in enumerate(self.criterions):
            losses.append(criterion(x, y) * self.criterion_weights[i])  # type: ignore
        if self.reduce == "mean":
            return sum(losses) / len(losses)
        if self.reduce == "sum":
            return sum(losses)
        if self.reduce is None:
            return losses
        raise ValueError(f"Invalid reduce: {self.reduce}")
