from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class GradRatioLoggingOptimizer:
    """Wraps optimizers that compute the ratio of the update to the parameter values."""
    def __init__(self, optimizer, name='training-optimizer'):
        self.__optimizer = optimizer
        self.__name = name
        self.__acc_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.__grad_ratio_acc_vars = OrderedDict()  # type: OrderedDict[str, tf.Variable]

    @property
    def optimizer(self):
        return self.__optimizer

    def print_ratios(self, session: tf.Session):
        count = self.__acc_count.eval(session) + 1e-10
        print('======================')
        print('Gradient Ratios')
        print('======================')
        for name, acc in self.__grad_ratio_acc_vars.items():
            print('%s: %.2e' % (name, acc.eval(session) / count))

        reset_ops = [tf.assign(self.__acc_count, 0)] + [tf.assign(v, 0) for v in self.__grad_ratio_acc_vars.values()]
        session.run(reset_ops)

    def minimize(self, loss):
        update_ops = [tf.assign_add(self.__acc_count, 1)]
        gradients_and_vars = self.__optimizer.compute_gradients(loss)
        for grad, var in gradients_and_vars:
            if grad is None:
                continue
            grad_ratio = tf.sqrt(tf.reduce_sum(tf.pow(grad, 2)) / tf.reduce_sum(tf.pow(var, 2)))
            ratio_acc_var = tf.Variable(0, trainable=False, dtype=tf.float32)
            self.__grad_ratio_acc_vars[var.name] = ratio_acc_var
            update_ops.append(tf.assign_add(ratio_acc_var, grad_ratio))
        grad_apply_op = self.__optimizer.apply_gradients(gradients_and_vars)
        update_ops.append(grad_apply_op)
        return control_flow_ops.group(*update_ops)