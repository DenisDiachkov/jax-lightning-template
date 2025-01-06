"""Cross-entropy loss."""

from typing import Callable

import equinox as eqnx
import optax  # pylint: disable=import[untyped]


class CELossWithIntegerLabels(eqnx.Module):
    """
    Cross-entropy loss.
    """

    _loss: Callable

    def __init__(self):
        self._loss = optax.softmax_cross_entropy_with_integer_labels

    def __call__(self, logits, labels):
        return self._loss(logits, labels)
