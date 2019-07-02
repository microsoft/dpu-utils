from typing import Optional, Callable

import tensorflow as tf

__all__ = [ 'get_activation' ]

def get_activation(activation_fun: Optional[str]) -> Optional[Callable]:
    if activation_fun is None:
        return None
    activation_fun = activation_fun.lower()
    if activation_fun == 'linear':
        return None
    if activation_fun == 'tanh':
        return tf.tanh
    if activation_fun == 'relu':
        return tf.nn.relu
    if activation_fun == 'leaky_relu':
        return tf.nn.leaky_relu
    if activation_fun == 'elu':
        return tf.nn.elu
    if activation_fun == 'selu':
        return tf.nn.selu
    if activation_fun == 'gelu':
        def gelu(input_tensor):
            cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
            return input_tensor * cdf
        return gelu
    else:
        raise ValueError("Unknown activation function '%s'!" % activation_fun)