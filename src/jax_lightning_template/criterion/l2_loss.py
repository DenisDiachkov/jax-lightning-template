"""Cross-entropy loss."""

from typing import Callable

import equinox as eqnx
import optax  # pylint: disable=import-untyped


class L2Loss(eqnx.Module):
    """
    Cross-entropy loss.
    """

    _loss: Callable

    def __init__(self):
        self._loss = optax.l2_loss

    def __call__(self, pred, gt):
        return self._loss(pred, gt)
