from typing import List

import numpy as np
import tensorflow as tf

from dpu_utils.tfutils import get_activation


class AsyncGGNN(object):
    @classmethod
    def default_params(cls):
        return {
            'hidden_size': 128,
            'edge_label_size': 16,

            'propagation_rounds': 4,  # Has to be an even number
            'propagation_substeps': 15,  # This is the maximal number of considered substeps

            'graph_rnn_cell': 'GRU',  # GRU or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU

            'use_edge_bias': False,
            'num_labeled_edge_types': 1,
            'num_unlabeled_edge_types': 4,
        }

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.num_labeled_edge_types = self.hyperparams['num_labeled_edge_types']
        self.num_unlabeled_edge_types = self.hyperparams['num_unlabeled_edge_types']
        self.num_edge_types = self.num_labeled_edge_types + self.num_unlabeled_edge_types

        self.__parameters = {}
        self.__make_parameters()

    @property
    def parameters(self):
        return self.__parameters

    def __make_parameters(self):
        activation_name = self.hyperparams['graph_rnn_activation'].lower()
        activation_fun = get_activation(activation_name)

        h_dim = self.hyperparams['hidden_size']
        e_dim = self.hyperparams['edge_label_size']
        self.__parameters['labeled_edge_weights'] = [tf.get_variable(name='labeled_edge_weights_typ%i' % e_typ,
                                                                     shape=[h_dim + e_dim, h_dim],
                                                                     initializer=tf.glorot_uniform_initializer())
                                                     for e_typ in range(self.num_labeled_edge_types)]
        self.__parameters['unlabeled_edge_weights'] = [tf.get_variable(name='unlabeled_edge_weights_typ%i' % e_typ,
                                                                       shape=[h_dim, h_dim],
                                                                       initializer=tf.glorot_uniform_initializer())
                                                       for e_typ in range(self.num_unlabeled_edge_types)]

        if self.hyperparams['use_edge_bias']:
            self.__parameters['labeled_edge_biases'] = [tf.Variable(np.zeros([h_dim], dtype=np.float32),
                                                                    name='labeled_edge_biases_typ%i' % e_typ)
                                                        for e_typ in range(self.num_labeled_edge_types)]
            self.__parameters['unlabeled_edge_biases'] = [tf.Variable(np.zeros([h_dim], dtype=np.float32),
                                                                      name='edge_biases_typ%i' % e_typ)
                                                          for e_typ in range(self.num_unlabeled_edge_types)]

        cell_type = self.hyperparams['graph_rnn_cell'].lower()
        if cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)

        self.__parameters['rnn_cell'] = cell

    def async_ggnn_layer(self,
                         initial_node_representation: tf.Tensor,
                         initial_nodes: List[tf.Tensor],
                         sending_nodes: List[List[List[tf.Tensor]]],
                         edge_labels: List[List[List[tf.Tensor]]],
                         msg_targets: List[List[tf.Tensor]],
                         receiving_nodes: List[List[tf.Tensor]],
                         receiving_node_num: List[tf.Tensor]) -> tf.Tensor:
        """
        Run through an async GGNN and return the representations of all nodes.
        :param initial_node_representation: the initial embeddings of the nodes.
                                            Shape: [-1, h_dim]
        :param initial_nodes: List of node id tensors I_{r}: Node IDs that will have no incoming edges in round r.
                              Inner Tensor Shape: [-1]
        :param sending_nodes: List of lists of lists of sending nodes S_{r,s,e}: Source node ids of edges of type e
                              propagating in step s of round r. By convention, 0..self.num_labeled_edges are labeled
                              edge types, and self.num_labeled_edges.. are unlabeled edge types.
                              Restrictions: If v in S_{r,s,e}, then v in R_{r,s'} for s' < s or v in I_{r}.
                              Inner Tensor Shape: [-1]
        :param edge_labels: List of lists of lists of (embeddings of) labels of edges L_{r,s,e}: Labels of edges of type
                            e propagating in step s of round r.
                            Restrictions: len(L_{r,s,e}) = len(S_{r,s,e})
                            Inner Tensor Shape: [-1, e_dim]
        :param msg_targets: List of lists of normalised edge target nodes T_{r,s}: Targets of edges propagating in step
                            s of round r, normalised to a continuous range starting from 0.
                            This is used for aggregating messages from the sending nodes.
                            Inner Tensor Shape: [-1]
        :param receiving_nodes: List of lists of receiving nodes R_{r,s}: Target node ids of aggregated messages in
                                propagation step s of round r.
                                Restrictions: If v in R_{r,s}, v not in R_{r,s'} for all s' != s and v not in I_{r}.
                                Inner Tensor Shape: [-1]
        :param receiving_node_num: Number of receiving nodes N_{r,s}
                                   Restrictions: N_{r,s} = len(R_{r,s})
                                   Inner Tensor Shape: [|Substeps|]
        :return: representations of all nodes after propagation according to schedule. Shape: [-1, h_dim]
        """
        with tf.variable_scope('async_ggnn'):
            cur_node_states = initial_node_representation

            for prop_round in range(self.hyperparams['propagation_rounds']):
                with tf.variable_scope('prop_round%i' % (prop_round,)):
                    # ---- Declare and fill tensor arrays used in tf.while_loop:
                    sending_nodes_ta = tf.TensorArray(
                        tf.int32,
                        infer_shape=False,
                        element_shape=[None],
                        size=self.hyperparams['propagation_substeps'] * self.num_edge_types,
                        name='sending_nodes'
                    )
                    edge_labels_ta = tf.TensorArray(
                        tf.float32,
                        infer_shape=False,
                        element_shape=[None, self.hyperparams['edge_label_size']],
                        size=self.hyperparams['propagation_substeps'] * self.num_labeled_edge_types,
                        name='edge_labels'
                    )
                    msg_targets_ta = tf.TensorArray(tf.int32,
                                                    infer_shape=False,
                                                    element_shape=[None],
                                                    size=self.hyperparams['propagation_substeps'],
                                                    name='msg_targets')
                    receiving_nodes_ta = tf.TensorArray(tf.int32,
                                                        infer_shape=False,
                                                        element_shape=[None],
                                                        size=self.hyperparams['propagation_substeps'],
                                                        clear_after_read=False,
                                                        name='receiving_nodes')
                    receiving_node_num_ta = tf.TensorArray(tf.int32,
                                                           infer_shape=False,
                                                           element_shape=[],
                                                           size=self.hyperparams['propagation_substeps'],
                                                           name='receiving_nodes_num')

                    for step in range(self.hyperparams['propagation_substeps']):
                        for labeled_edge_typ in range(self.num_labeled_edge_types):
                            sending_nodes_ta = sending_nodes_ta.write(step * self.num_edge_types + labeled_edge_typ,
                                                                      sending_nodes[prop_round][step][labeled_edge_typ])
                            edge_labels_ta = edge_labels_ta.write(step * self.num_labeled_edge_types + labeled_edge_typ,
                                                                  edge_labels[prop_round][step][labeled_edge_typ])
                        for unlabeled_edge_typ in range(self.num_unlabeled_edge_types):
                            shifted_edge_typ = self.num_labeled_edge_types + unlabeled_edge_typ
                            sending_nodes_ta = sending_nodes_ta.write(step * self.num_edge_types + shifted_edge_typ,
                                                                      sending_nodes[prop_round][step][shifted_edge_typ])
                        msg_targets_ta = msg_targets_ta.write(step, msg_targets[prop_round][step])
                        receiving_nodes_ta = receiving_nodes_ta.write(step, receiving_nodes[prop_round][step])
                    receiving_node_num_ta = receiving_node_num_ta.unstack(receiving_node_num[prop_round])

                    new_node_states_ta = tf.TensorArray(tf.float32,
                                                        infer_shape=False,
                                                        element_shape=[self.hyperparams['hidden_size']],
                                                        size=tf.shape(cur_node_states)[0],
                                                        clear_after_read=False,
                                                        name='new_node_states')

                    # ---- Actual propagation schedule implementation:
                    # Initialize the initial nodes with their state from last round:
                    new_node_states_ta = new_node_states_ta.scatter(initial_nodes[prop_round],
                                                                    tf.gather(cur_node_states, initial_nodes[prop_round]))

                    def do_substep(substep_id, new_node_states_ta):
                        # For each edge active in this substep, pull source state and transform:
                        sending_states_per_edge_type = []
                        edge_labels_per_type = []
                        for labeled_edge_typ in range(self.num_labeled_edge_types):
                            sending_states_per_edge_type.append(
                                new_node_states_ta.gather(sending_nodes_ta.read(
                                    substep_id * self.num_edge_types + labeled_edge_typ
                                ))
                            )
                            edge_labels_per_type.append(edge_labels_ta.read(
                                substep_id * self.num_labeled_edge_types + labeled_edge_typ
                            ))
                        for unlabeled_edge_typ in range(self.num_unlabeled_edge_types):
                            shifted_edge_typ = self.num_labeled_edge_types + unlabeled_edge_typ
                            sending_states_per_edge_type.append(new_node_states_ta.gather(
                                sending_nodes_ta.read(substep_id * self.num_edge_types + shifted_edge_typ)
                            ))

                        # Collect old states for receiving nodes
                        substep_receiving_nodes = receiving_nodes_ta.read(substep_id)
                        old_receiving_node_states = tf.gather(cur_node_states, substep_receiving_nodes)
                        old_receiving_node_states.set_shape([None, self.hyperparams['hidden_size']])

                        msg_targets_this_step = msg_targets_ta.read(substep_id)
                        receiving_node_num_this_step = receiving_node_num_ta.read(substep_id)

                        substep_new_node_states = self.propagate_one_step(
                            sending_states_per_edge_type, edge_labels_per_type,
                            msg_targets_this_step, receiving_node_num_this_step,
                            old_receiving_node_states
                        )

                        # Write updated states back:
                        new_node_states_ta = new_node_states_ta.scatter(indices=substep_receiving_nodes,
                                                                        value=substep_new_node_states,
                                                                        name="state_scatter_round%i" % (prop_round,))
                        return substep_id + 1, new_node_states_ta

                    def is_done(substep_id, new_node_states_ta_unused):
                        return tf.logical_and(substep_id < self.hyperparams['propagation_substeps'],
                                              tf.greater(tf.shape(receiving_nodes_ta.read(substep_id))[0], 0))

                    _, new_node_states_ta = tf.while_loop(cond=is_done,
                                                          body=do_substep,
                                                          loop_vars=[tf.constant(0), new_node_states_ta]
                                                         )

                    cur_node_states = new_node_states_ta.stack(name="state_stack_round%i" % (prop_round,))

            return cur_node_states

    def propagate_one_step(self,
                           sending_states_per_edge_type: List[tf.Tensor],
                           edge_labels_per_type: List[tf.Tensor],
                           msg_targets_this_step: tf.Tensor,
                           receiving_node_num_this_step: tf.Tensor,
                           old_receiving_node_states: tf.Tensor) -> tf.Tensor:
        sent_messages = []
        for (edge_typ, sending_state_representations) in enumerate(sending_states_per_edge_type):
            if edge_typ < self.num_labeled_edge_types:

                messages = tf.matmul(tf.concat([sending_state_representations, edge_labels_per_type[edge_typ]],
                                               axis=-1),
                                     self.__parameters['labeled_edge_weights'][edge_typ])
                if self.hyperparams['use_edge_bias']:
                    messages += self.__parameters['labeled_edge_biases'][edge_typ]
            else:
                shifted_edge_typ = edge_typ - self.num_labeled_edge_types
                messages = tf.matmul(
                    sending_state_representations, self.__parameters['unlabeled_edge_weights'][shifted_edge_typ]
                )
                if self.hyperparams['use_edge_bias']:
                    messages += self.__parameters['unlabeled_edge_biases'][shifted_edge_typ]
            sent_messages.append(messages)

        # Stack all edge messages and aggregate as sum for each receiving node:
        sent_messages = tf.concat(sent_messages, axis=0)
        aggregated_received_messages = tf.unsorted_segment_sum(
            sent_messages, msg_targets_this_step, receiving_node_num_this_step
        )

        # Combine old states in RNN cell with incoming messages
        aggregated_received_messages.set_shape([None, self.hyperparams['hidden_size']])
        new_node_states = self.__parameters['rnn_cell'](aggregated_received_messages,
                                                        old_receiving_node_states)[1]
        return new_node_states
