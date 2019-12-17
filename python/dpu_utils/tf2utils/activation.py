"""Custom activation functions."""
from typing import Optional, Callable
import math as m

import tensorflow as tf


def gelu(input_tensor: tf.Tensor) -> tf.Tensor:
    """An approximation to the GELU activation function as used in the paper
    https://arxiv.org/pdf/1810.04805.pdf
    """
    cdf = 0.5 * (
        1.0
        + tf.tanh(
            (tf.sqrt(2 / m.pi) * (input_tensor + 0.044715 * tf.pow(input_tensor, 3)))
        )
    )
    return input_tensor * cdf


def get_activation_function_by_name(
    activation_fn_name: Optional[str],
) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
    """Convert from an activation function name to the function itself."""
    if activation_fn_name is None:
        return None
    activation_fn_name = activation_fn_name.lower()

    string_to_activation_fn = {
        "linear": None,
        "tanh": tf.nn.tanh,
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "gelu": gelu,
    }
    activation_fn = string_to_activation_fn.get(activation_fn_name)
    if activation_fn is None:
        raise ValueError(f"Unknown activation function: {activation_fn_name}")
    return activation_fn
