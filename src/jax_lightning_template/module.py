"""
Module for the LightningModule
"""

from functools import partial
from typing import Any, Dict, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # pylint: disable=import
from pytorch_lightning import LightningModule

from .utils import utils


# pylint: disable=too-many-ancestors
class JaxLightningModule(LightningModule):
    """
    Base class for the LightningModule.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model: str,
        model_params: dict,
        optimizer: str | None = None,
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        criterion: str | None = None,
        criterion_params: dict | None = None,
        log_metrics_every_n_steps: int = 1,
        visualization_every_n_steps: int = 30,
        logging_function: str | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if logging_function is not None:
            self.logging_function = utils.get_cls(logging_function)
        else:
            self.logging_function = lambda *args: None

        self.automatic_optimization = False
        # pylint: disable=assignment-from-no-return
        model, state = eqx.nn.make_with_state(utils.get_cls(model))(**model_params)
        self.model: eqx.Module = cast(eqx.Module, model)
        self.state: eqx.nn.State = cast(eqx.nn.State, state)
        self.key = jax.random.PRNGKey(seed)
        if lr_scheduler is not None:
            self.lr_scheduler = utils.get_instance(
                lr_scheduler,
                lr_scheduler_params or {},
            )
        if optimizer is not None:
            self.optimizer: optax.GradientTransformation = utils.get_instance(
                optimizer,
                (optimizer_params or {})
                | ({"learning_rate": self.lr_scheduler} if lr_scheduler else {}),
            )
            self.optimizer_state: optax.OptState = self.optimizer.init(
                eqx.filter(model, eqx.is_inexact_array)
            )
        if criterion is not None:
            self.criterion = utils.get_instance(criterion, criterion_params or {})

        self.log_metrics_every_n_steps = log_metrics_every_n_steps
        self.visualization_every_n_steps = visualization_every_n_steps

    def log_metrics(self, *args, **kwargs) -> None:
        """
        This function logs the metrics.
        """

    def visualization(self, *args, **kwargs) -> None:
        """
        This function visualizes the results.
        """

    def log_all(
        self,
        mode: Literal["train", "val", "test"],
        batch: Any,
        loss: Any,
        batch_idx: int,
        **kwargs,
    ) -> None:
        """
        This function logs all the metrics.
        """

        # Log the loss
        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input"].shape[0],
        )

        # Log the learning rate
        # if hasattr(self, "lr_scheduler"):
        #     self.log(
        #         #  pylint: disable=unsubscriptable-object
        #         "lr", self.optimizer_state.hyperparams['learning_rate'], prog_bar=True, sync_dist=True
        #     )

        # Log the metrics
        if (batch_idx + 1) % self.log_metrics_every_n_steps == 0:
            self.log_metrics(batch)

        # Visualize the results
        if (batch_idx + 1) % self.visualization_every_n_steps == 0:
            self.visualization(batch)

    def save_test_results(self, batch: Any, batch_idx: int) -> None:
        """
        This function saves the test results.
        """

    @staticmethod
    @eqx.filter_jit
    def eqx_batch_forward(model, state, x, key, inference: bool = True):
        return eqx.filter_vmap(
            eqx.nn.inference_mode(model) if inference else model,
            in_axes=(0, None, None),
            out_axes=(0, None),
            axis_name="batch",
        )(x, state, key)

    @staticmethod
    @eqx.filter_jit
    def eqx_batch_loss(output, target, criterion):
        return eqx.filter_vmap(criterion, in_axes=(0, 0), axis_name="batch")(
            output, target
        )

    @staticmethod
    @eqx.filter_jit
    def eqx_make_step(
        model: eqx.Module,
        state: eqx.nn.State,
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        criterion: eqx.Module,
        batch: Dict[str, Any],
        key: jnp.ndarray,
    ) -> tuple[
        eqx.Module, eqx.nn.State, optax.OptState, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """
        This function makes a step.
        """

        def loss_fn(model, state, x, y, criterion, key):
            output, state = JaxLightningModule.eqx_batch_forward(
                model, state, x, key, False
            )
            loss = JaxLightningModule.eqx_batch_loss(output, y, criterion)
            return loss.mean(), (output, state)

        (loss, (output, state)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model, state, batch["input"], batch["target"], criterion, key)
        # Gradient clipping
        grads = jax.tree.map(lambda x: jnp.clip(x, -1, 1), grads)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        model = eqx.apply_updates(model, updates)
        return model, state, optimizer_state, loss, output, grads

    # pylint: disable=arguments-differ
    def training_step(self, batch, batch_idx: int):
        (
            self.model,
            self.state,
            self.optimizer_state,
            loss,
            batch["output"],
            _,
        ) = JaxLightningModule.eqx_make_step(
            self.model,
            self.state,
            self.optimizer_state,
            self.optimizer,
            self.criterion,
            batch,
            self.key,
        )
        self.key = jax.random.split(self.key)[0]
        self.log_all("train", batch, jax.device_get(loss).mean(), batch_idx)
        if self.logging_function is not None:
            self.logging_function(batch, batch_idx, batch["output"])

    # pylint: disable=arguments-differ, unused-argument
    def validation_step(self, batch, batch_idx):
        batch["output"], self.state = JaxLightningModule.eqx_batch_forward(
            self.model, self.state, batch["input"], self.key, True
        )
        self.key = jax.random.split(self.key)[0]
        loss = JaxLightningModule.eqx_batch_loss(
            batch["output"], batch["target"], self.criterion
        )
        self.log_all("val", batch, jax.device_get(loss).mean(), batch_idx)

    # pylint: disable=arguments-differ, unused-argument
    def test_step(self, batch, batch_idx):
        batch["output"], self.state = JaxLightningModule.eqx_batch_forward(
            self.model, self.state, batch["input"], self.key, True
        )
        self.key = jax.random.split(self.key)[0]
        loss = JaxLightningModule.eqx_batch_loss(
            batch["output"], batch["target"], self.criterion
        )
        self.log_all("test", batch, jax.device_get(loss).mean(), batch_idx)
        self.save_test_results(batch, batch_idx)

    def configure_optimizers(self):
        pass
