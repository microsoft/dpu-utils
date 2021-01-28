"""MLP layer."""
import sys
from typing import Callable, List, Optional, Union

import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        out_size: int,
        hidden_layers: Union[List[int], int] = 1,
        use_biases: bool = False,
        activation_fun: Optional[Callable[[tf.Tensor], tf.Tensor]] = tf.nn.relu,
        dropout_rate: float = 0.0,
        name: str = "MLP",
    ):
        """
        Create new MLP with given number of hidden layers.

        Arguments:
            out_size: Dimensionality of output.
            hidden_layers: Either an integer determining number of hidden layers, which will have
                out_size units each; or list of integers whose lengths determines the number of
                hidden layers and whose contents the number of units in each layer.
            use_biases: Flag indicating use of bias in fully connected layers.
            activation_fun: Activation function applied between hidden layers (NB: the output of the
                MLP is always the direct result of a linear transformation)
            dropout_rate: Dropout applied to inputs of each MLP layer.
            name: Name of the MLP, used in names of created variables.
        """
        super().__init__()
        if isinstance(hidden_layers, int):
            if out_size == 1:
                print(
                    f"W: In {name}, was asked to use {hidden_layers} layers of size 1, which is most likely wrong."
                    f" Switching to {hidden_layers} layers of size 32; to get hidden layers of size 1,"
                    f" use hidden_layers=[1,...,1] explicitly.",
                    file=sys.stderr,
                )
                self._hidden_layer_sizes = [32] * hidden_layers
            else:
                self._hidden_layer_sizes = [out_size] * hidden_layers
        else:
            self._hidden_layer_sizes = hidden_layers

        if len(self._hidden_layer_sizes) > 1:
            assert (
                activation_fun is not None
            ), "Multiple linear layers without an activation"

        self._out_size = out_size
        self._use_biases = use_biases
        self._activation_fun = activation_fun
        self._dropout_rate = dropout_rate
        self._layers = []  # type: List[tf.keras.layers.Dense]
        self._name = name

    def build(self, input_shape):
        last_shape_dim = input_shape[-1]
        for hidden_layer_idx, hidden_layer_size in enumerate(self._hidden_layer_sizes):
            with tf.name_scope(f"{self._name}_dense_layer_{hidden_layer_idx}"):
                self._layers.append(
                    tf.keras.layers.Dense(
                        units=hidden_layer_size,
                        use_bias=self._use_biases,
                        activation=self._activation_fun,
                        name=f"{self._name}_dense_layer_{hidden_layer_idx}",
                    )
                )
                self._layers[-1].build(tf.TensorShape(input_shape[:-1] + [last_shape_dim]))
                last_shape_dim = hidden_layer_size

        # Output layer:
        with tf.name_scope(f"{self._name}_final_layer"):
            self._layers.append(
                tf.keras.layers.Dense(
                    units=self._out_size,
                    use_bias=self._use_biases,
                    name=f"{self._name}_final_layer",
                )
            )
            self._layers[-1].build(tf.TensorShape(input_shape[:-1] + [last_shape_dim]))

        super().build(input_shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, input: tf.Tensor, training: bool) -> tf.Tensor:
        activations = input
        for layer in self._layers[:-1]:
            if training:
                activations = tf.nn.dropout(activations, rate=self._dropout_rate)
            activations = layer(activations)
        return self._layers[-1](activations)
