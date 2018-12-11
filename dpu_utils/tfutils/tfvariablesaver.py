import tensorflow as tf
from typing import Callable, Dict, Optional
import numpy as np


class TFVariableSaver:
    """
    Save all variables in the graph and restore them, in a way that the values are serializable by pickle.
    """
    def __init__(self):
        self.__saved_variables = {}  # type: Dict[str, np.ndarray]

    def save_all(self, session: tf.Session, exclude_variable: Optional[Callable[[str], bool]]=None) -> None:
        self.__saved_variables = {}
        for variable in session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in self.__saved_variables
            if exclude_variable is not None and exclude_variable(variable.name):
                continue
            self.__saved_variables[variable.name] = variable.value().eval()

    def has_saved_variables(self) -> bool:
        return len(self.__saved_variables) > 0

    def restore_saved_values(self, session: tf.Session) -> None:
        assert len(self.__saved_variables) > 0
        save_ops = []
        with tf.name_scope("restore"):
            for variable in session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if variable.name in self.__saved_variables:
                    saved_value = self.__saved_variables[variable.name]
                    if len(variable.shape) == 0 or variable.shape[0]._value == saved_value.shape[0]:  # Scalars or the size hasn't changed.
                        save_ops.append(variable.assign(saved_value))
                    else:
                        # Allow expanding saved variables
                        print('Store value for %s has shape %s but the variable has shape %s. Padding with zeros.'
                              % (variable.name, saved_value.shape, variable.shape))

                        initial_value = np.zeros([variable.shape[i]._value for i in range(len(variable.shape))],
                                                 dtype=variable.dtype.as_numpy_dtype)
                        initial_value[:saved_value.shape[0]] = saved_value
                        save_ops.append(variable.assign(initial_value))
                else:
                    print('Initializing %s from random since no saved value was found.' % variable.name)
                    save_ops.append(tf.variables_initializer([variable]))
        session.run(save_ops)
