from typing import List, Optional, Dict, Any
from collections import namedtuple

import tensorflow as tf

from dpu_utils.tfutils import unsorted_segment_log_softmax

from dpu_utils.tfutils import get_activation

SMALL_NUMBER = 1e-7


GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',
                                         'edge_feature_gate_weights',
                                         'edge_feature_gate_bias'])


class SparseGGNN:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.num_edge_types = self.params['n_edge_types']
        assert self.num_edge_types > 0, 'GNN should have at least one edge type'
        h_dim = self.params['hidden_size']

        edge_feature_sizes = self.params.get('edge_features_size', {})  # type: Dict[int, int]

        if self.params['add_backwards_edges']:
            effective_num_edge_types = self.num_edge_types * 2
        else:
            effective_num_edge_types = self.num_edge_types

        message_aggregation_type = self.params.get('message_aggregation', 'sum')
        if message_aggregation_type == 'sum':
            self.unsorted_segment_aggregation_func = tf.unsorted_segment_sum
        elif message_aggregation_type == 'max':
            self.unsorted_segment_aggregation_func = tf.unsorted_segment_max
        else:
            raise Exception('Unrecognized message_aggregation type %s' % message_aggregation_type)

        # Generate per-layer values for edge weights, biases and gated units. If we tie them, they are just copies:
        self.__weights = GGNNWeights([], [], [], [], [], [])
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.get_variable(name='gnn_edge_weights',
                                               shape=[effective_num_edge_types * h_dim, h_dim],
                                               initializer=tf.glorot_normal_initializer())
                edge_weights = tf.reshape(edge_weights, [effective_num_edge_types, h_dim, h_dim])
                self.__weights.edge_weights.append(edge_weights)

                if self.params['use_propagation_attention']:
                    self.__weights.edge_type_attention_weights.append(tf.get_variable(name='edge_type_attention_weights',
                                                                                      shape=[effective_num_edge_types],
                                                                                      initializer=tf.ones_initializer()))

                self.__weights.edge_feature_gate_weights.append({})
                self.__weights.edge_feature_gate_bias.append({})
                for edge_type, edge_feature_size in edge_feature_sizes.items():
                    self.__weights.edge_feature_gate_weights[layer_idx][edge_type] = \
                        tf.get_variable(name='gnn_edge_%i_feature_gate_weights' % (edge_type,),
                                        shape=[2 * edge_feature_size, 1],
                                        initializer=tf.ones_initializer())
                    self.__weights.edge_feature_gate_bias[layer_idx][edge_type] = \
                        tf.get_variable(name='gnn_edge_%i_feature_gate_bias' % (edge_type,),
                                        shape=[1],
                                        initializer=tf.zeros_initializer())
                    if self.params['add_backwards_edges']:
                        self.__weights.edge_feature_gate_weights[layer_idx][self.num_edge_types + edge_type] = \
                            tf.get_variable(name='gnn_edge_%i_feature_gate_weights' % (self.num_edge_types + edge_type,),
                                            shape=[2 * edge_feature_size, 1],
                                            initializer=tf.ones_initializer())
                        self.__weights.edge_feature_gate_bias[layer_idx][self.num_edge_types + edge_type] = \
                            tf.get_variable(name='gnn_edge_%i_feature_gate_bias' % (self.num_edge_types + edge_type,),
                                            shape=[1],
                                            initializer=tf.zeros_initializer())

                if self.params['use_edge_bias']:
                    self.__weights.edge_biases.append(tf.get_variable(name='gnn_edge_biases',
                                                                      shape=[effective_num_edge_types, h_dim],
                                                                      initializer=tf.zeros_initializer()))

                cell = self.__create_rnn_cell(h_dim)
                self.__weights.rnn_cells.append(cell)

    def __create_rnn_cell(self, h_dim: int):
        activation_name = self.params['graph_rnn_activation'].lower()
        activation_fun = get_activation(activation_name)

        cell_type = self.params['graph_rnn_cell'].lower()
        if cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
        elif cell_type == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
        else:
            raise Exception("Unknown RNN cell type '%s'." % cell_type)
        return cell

    def sparse_gnn_layer(self,
                         dropout_keep_rate: tf.Tensor,
                         node_embeddings: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
                         num_incoming_edges_per_type: Optional[tf.Tensor],
                         num_outgoing_edges_per_type: Optional[tf.Tensor],
                         edge_features: Dict[int, tf.Tensor]) -> tf.Tensor:
        """
        Run through a GNN and return the representations of the nodes.
        :param dropout_keep_rate: See name.
        :param node_embeddings: the initial embeddings of the nodes.
        :param adjacency_lists: a list of *sorted* adjacency indexes per edge type
        :param num_incoming_edges_per_type: [v, num_edge_types] tensor indicating number of incoming edges per type.
                                            Required if use_edge_bias or use_edge_msg_avg_aggregation is true.
        :param num_outgoing_edges_per_type: [v, num_edge_types] tensor indicating number of incoming edges per type.
                                            Required if add_backwards_edges and (use_edge_bias or use_edge_msg_avg_aggregation) is true.
        :param edge_features: a dictionary of edge_type -> num_edges x feature_length for the edges that have features.
        :return: the representations of the nodes
        """
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)
        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]

        # Note that we optionally support adding (implicit) backwards edges. If turned on, we introduce additional
        # edge type indices [self.num_edge_types .. 2*self.num_edge_types - 1], with their own weights.

        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        if self.params['add_backwards_edges']:
            for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
                edge_targets = adjacency_list_for_edge_type[:, 0]
                message_targets.append(edge_targets)
                message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * (self.num_edge_types + edge_type_idx ))
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        with tf.variable_scope('gnn_scope'):
            node_states_per_layer = []  # list of tensors of shape [V, D], one entry per layer (the final state of that layer)
            node_states_per_layer.append(node_embeddings)
            num_nodes = tf.shape(node_embeddings, out_type=tf.int32)[0]

            for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
                with tf.variable_scope('gnn_layer_%i' % layer_idx):
                    # Extract residual messages, if any:
                    layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                    if layer_residual_connections is None:
                        layer_residual_states = []
                    else:
                        layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                                 for residual_layer_idx in layer_residual_connections]

                    if self.params['use_propagation_attention']:
                        message_edge_type_factors = tf.nn.embedding_lookup(params=self.__weights.edge_type_attention_weights[layer_idx],
                                                                           ids=message_edge_types)  # Shape [M]

                    # Record new states for this layer. Initialised to last state, but will be updated below:
                    node_states_per_layer.append(node_states_per_layer[-1])

                    for step in range(num_timesteps):
                        with tf.variable_scope('timestep_%i' % step):
                            messages = []  # list of tensors of messages of shape [E, D]
                            message_source_states = []  # list of tensors of edge source states of shape [E, D]

                            # Collect incoming messages per edge type
                            def compute_messages_for_edge_type(data_edge_type_idx: int, weights_edge_type_idx: int, edge_sources: tf.Tensor) -> None:
                                edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                            ids=edge_sources)  # Shape [E, D]
                                edge_weights = tf.nn.dropout(self.__weights.edge_weights[layer_idx][weights_edge_type_idx],
                                                             rate=1-dropout_keep_rate)
                                all_messages_for_edge_type = tf.matmul(edge_source_states, edge_weights)  # Shape [E, D]

                                if data_edge_type_idx in edge_features:
                                    edge_feature_augmented = tf.concat([edge_features[data_edge_type_idx],
                                                                        1 / (edge_features[data_edge_type_idx] + SMALL_NUMBER)],
                                                                       axis=-1)  # Shape [E, 2*edge_size]
                                    all_messages_gate_value = \
                                        tf.sigmoid(self.__weights.edge_feature_gate_bias[layer_idx][weights_edge_type_idx]
                                                   + tf.matmul(edge_feature_augmented,
                                                               self.__weights.edge_feature_gate_weights[layer_idx][weights_edge_type_idx]))  # Shape [E, 1]
                                    all_messages_for_edge_type = all_messages_gate_value * all_messages_for_edge_type

                                messages.append(all_messages_for_edge_type)
                                message_source_states.append(edge_source_states)

                            for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
                                compute_messages_for_edge_type(edge_type_idx, edge_type_idx, adjacency_list_for_edge_type[:, 0])
                            if self.params['add_backwards_edges']:
                                for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
                                    compute_messages_for_edge_type(edge_type_idx, self.num_edge_types + edge_type_idx, adjacency_list_for_edge_type[:, 1])

                            messages = tf.concat(messages, axis=0)  # Shape [M, D]

                            if self.params['use_propagation_attention']:
                                message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                                message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                               ids=message_targets)  # Shape [M, D]
                                message_attention_scores = tf.einsum('mi,mi->m', message_source_states, message_target_states)  # Shape [M]
                                message_attention_scores = message_attention_scores * message_edge_type_factors

                                message_log_attention = unsorted_segment_log_softmax(logits=message_attention_scores,
                                                                                 segment_ids=message_targets,
                                                                                 num_segments=num_nodes)

                                message_attention = tf.exp(message_log_attention) # Shape [M]
                                # Step (4): Weight messages using the attention prob:
                                messages = messages * tf.expand_dims(message_attention, -1)

                            incoming_messages = self.unsorted_segment_aggregation_func(data=messages,
                                                                                  segment_ids=message_targets,
                                                                                  num_segments=num_nodes)  # Shape [V, D]

                            if self.params['use_edge_bias']:
                                incoming_messages += tf.matmul(num_incoming_edges_per_type,
                                                               self.__weights.edge_biases[layer_idx][0:self.num_edge_types])  # Shape [V, D]
                                if self.params['add_backwards_edges']:
                                    incoming_messages += tf.matmul(num_outgoing_edges_per_type,
                                                                   self.__weights.edge_biases[layer_idx][self.num_edge_types:])  # Shape [V, D]

                            if self.params['use_edge_msg_avg_aggregation']:
                                num_incoming_edges = tf.reduce_sum(num_incoming_edges_per_type,
                                                                   keep_dims=True, axis=-1)  # Shape [V, 1]
                                if self.params['add_backwards_edges']:
                                    num_incoming_edges += tf.reduce_sum(num_outgoing_edges_per_type,
                                                                        keep_dims=True, axis=-1)  # Shape [V, 1]
                                incoming_messages /= num_incoming_edges + SMALL_NUMBER

                            incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                             axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                            # pass updated vertex features into RNN cell
                            node_states_per_layer[-1] = self.__weights.rnn_cells[layer_idx](incoming_information,
                                                                                            node_states_per_layer[-1])[1]  # Shape [V, D]

        return node_states_per_layer[-1]
