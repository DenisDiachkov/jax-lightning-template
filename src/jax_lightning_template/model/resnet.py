"""
ResNet model.
"""

from typing import Any

import equinox
import jax


class ResNetBlock(equinox.Module):
    """
    ResNet block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the kernel.
        padding (int): Padding of the kernel
    """

    conv1: equinox.nn.Conv2d
    bn1: equinox.nn.BatchNorm
    conv2: equinox.nn.Conv2d
    bn2: equinox.nn.BatchNorm

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        key: Any,
    ):
        super().__init__()
        _, subkey = jax.random.split(key)
        self.conv1 = equinox.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, key=subkey
        )
        self.bn1 = equinox.nn.BatchNorm(out_channels, axis_name="batch", momentum=0.9)
        _, subkey = jax.random.split(subkey)
        self.conv2 = equinox.nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding, key=subkey
        )
        self.bn2 = equinox.nn.BatchNorm(out_channels, axis_name="batch", momentum=0.9)

    def __call__(self, x, state, key):
        """
        Forward pass.
        """
        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)
        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out += x
        out = jax.nn.relu(out)
        return out, state


class ResNet(equinox.Module):
    """
    ResNet model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the kernel.
        padding (int): Padding of the kernel.
        num_classes (int): Number of classes.
        num_blocks (int): Number of blocks
    """

    conv: equinox.nn.Conv2d
    bn: equinox.nn.BatchNorm
    resnet_blocks: equinox.nn.Sequential
    adaptive_avg_pool2d: equinox.nn.AdaptivePool
    fc: equinox.nn.Linear

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_classes: int = 10,
        num_blocks: int = 3,
        seed: int = 0,
    ):
        super().__init__()
        _, *subkeys = jax.random.split(jax.random.PRNGKey(seed), num_blocks + 2)
        self.conv = equinox.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, key=subkeys[-1]
        )
        self.bn = equinox.nn.BatchNorm(out_channels, axis_name="batch", momentum=0.9)
        self.resnet_blocks = equinox.nn.Sequential(
            layers=tuple(
                ResNetBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    key=subkeys[i],
                )
                for i in range(num_blocks)
            )
        )
        self.adaptive_avg_pool2d = equinox.nn.AdaptivePool(
            (1, 1), 2, operation=jax.numpy.mean
        )
        self.fc = equinox.nn.Linear(out_channels, num_classes, key=subkeys[-2])

    def __call__(self, x, state, key):
        """
        Forward pass.
        """

        out = self.conv(x)
        out, state = self.bn(out, state)
        out = jax.nn.relu(out)
        for block in self.resnet_blocks.layers:
            out, state = block(out, state, key)
        # Adaptive average pooling
        out = self.adaptive_avg_pool2d(out)
        out = out.flatten()
        out = self.fc(out)
        return out, state
