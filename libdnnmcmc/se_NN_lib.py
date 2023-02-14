'''
@author: Andreas Bott

custom layers, losses and metrics for Neural Networks in steady-state DH system modelling
'''


import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions

#%% Custom Scaling Layers:
class MyScalingLayer(keras.layers.Layer):
    def __init__(self, offset=None, scaling=None, mapping_matrix=None, mapping_matrix_ind=None,
                 mapping_matrix_dense_shape=None, name='MyScalingLayer'):
        '''
        output = offset + mapping_matrix @ scaling * inputs

        '''
        super().__init__(name=name)
        if offset is not None:
            self.offset = tf.Variable(offset, trainable=False)
        if scaling is not None:
            self.scaling = tf.Variable(scaling, trainable=False)
        if mapping_matrix is None and mapping_matrix_ind is None:
            # default behaviour - one to one mapping
            self.mapping_matrix = tf.sparse.eye(offset.get_shape()[0])
        if mapping_matrix is not None:
            self.mapping_matrix = mapping_matrix
        elif mapping_matrix_ind is not None:
            self.mapping_matrix = tf.sparse.SparseTensor(indices=mapping_matrix_ind,
                                                         values=tf.ones(tf.shape(mapping_matrix_ind)[0],
                                                                        dtype=tf.float64),
                                                         dense_shape=mapping_matrix_dense_shape)
            self.offset = tf.Variable(tf.ones((mapping_matrix_dense_shape[0], 1), dtype=tf.float64), trainable=False)
            self.scaling = tf.Variable(tf.ones(tf.shape(mapping_matrix_ind)[0], dtype=tf.float64), trainable=False)


    def call(self, inputs, *args, **keyargs):
        return tf.transpose(tf.math.add(self.offset,
                                        tf.sparse.sparse_dense_matmul(self.mapping_matrix.with_values(self.scaling),
                                                                      inputs, adjoint_b=True)))

    def get_config(self):
        return {
            'name': self.name,
            'mapping_matrix_ind': tf.cast(self.mapping_matrix.indices, tf.int64).numpy(),
            'mapping_matrix_dense_shape': tf.cast(self.mapping_matrix.shape, tf.int64).numpy()
            }


#%% loss functions:
@tf.function
def loss_SE(SE, T, mf, p, T_end):
    # calculations are way easier to get consistent if the batch dimension is last:
    loss = SE.evaluate_state_equations('forwardpass',
                                       T=tf.transpose(T), mf=tf.transpose(mf), p=tf.transpose(p),
                                       T_end=tf.transpose(T_end))
    return tf.reduce_sum(loss ** 2, axis=0)


@tf.function
def loss_mape(y_true, y_pred):
    return 100. * tf.reduce_mean(tf.math.abs((y_true - y_pred) / y_true))


#%% Loss Classes based on the loss functions
class LossWeightedMSE(keras.losses.Loss):
    def __init__(self, n_nodes, n_edges, lambda_T=1, lambda_mf=1, lambda_p=1, lambda_Tend=1, name='weighted_MES'):
        super().__init__(name=name)
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.lambda_T = lambda_T
        self.lambda_mf = lambda_mf
        self.lambda_p = lambda_p
        self.lambda_Tend = lambda_Tend
        weighting_vector = np.zeros((2*n_edges+2*n_nodes))
        weighting_vector[0: n_nodes] = lambda_T
        weighting_vector[n_nodes: n_nodes+n_edges] = lambda_mf
        weighting_vector[n_nodes+n_edges: 2*n_nodes+n_edges] = lambda_p
        weighting_vector[2*n_nodes+n_edges: 2*n_nodes+2*n_edges] = lambda_Tend
        self.weighting_vector = tf.constant(weighting_vector)

    def call(self, y_true, y_pred):
        """ calculates the mean squared distance between y_true and y_pred, weighting T, mf, p, Tend with cor. lambda"""
        return tf.reduce_mean((self.weighting_vector*(y_true - y_pred))**2, axis=1)

    def get_config(self):
        return{'n_nodes': self.n_nodes,
               'n_edges': self.n_edges,
               'lambda_T': self.lambda_T,
               'lambda_mf': self.lambda_mf,
               'lambda_p': self.lambda_p,
               'lambda_Tend': self.lambda_Tend,
               'name': self.name}

#%% Metric Classes
class MetricMAPE_T(keras.metrics.Metric):
    def __init__(self, n_nodes, n_edges, name='MAPE_T', **kwargs):
        super().__init__(name=name)
        self.lb = 0
        self.ub = n_nodes
        self.value = tf.Variable(0.0, dtype=tf.float64, name='Mape_T_loss_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(0.0)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }

class MetricMAPE_mf(keras.metrics.Metric):
    def __init__(self, n_nodes, n_edges, name='MAPE_mf', **kwargs):
        super().__init__(name=name)
        self.lb = n_nodes
        self.ub = n_nodes + n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='Mape_mf_loss_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(0.0)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAPE_p(keras.metrics.Metric):
    def __init__(self, n_nodes, n_edges, name='MAPE_p', **kwargs):
        super().__init__(name=name)
        self.lb = n_nodes + n_edges
        self.ub = 2 * n_nodes + n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='Mape_p_loss_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(0.0)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }

class MetricMAPE_Tend(keras.metrics.Metric):
    def __init__(self, n_nodes, n_edges, name='MAPE_Tend', **kwargs):
        super().__init__(name=name)
        self.lb = 2 * n_nodes + n_edges
        self.ub = 2 * n_nodes + 2 * n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='Mape_Tend_loss_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(0.0)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAE_T(keras.metrics.Metric):
    def __init__(self, n_nodes, n_edges, name='MAE_T', **kwargs):
        super().__init__(name=name)
        self.lb = 0
        self.ub = n_nodes
        self.value = tf.Variable(0.0, dtype=tf.float64, name='Mae_T_loss_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mae(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(0.0)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }

#%% plot functions:
def plot_history(history):
    """
    :param history:  keras.history:    training history of the DNN
    :return: None

    this function plots the losses on the training and validation set for the metrics defined above.
    """
    epochs = history.epoch
    keys = list(history.history.keys())
    fig_train, ax_train = plt.subplots()
    fig_val, ax_val = plt.subplots()

    for key in keys:
        values = history.history[key]
        label = ''
        if 'loss' == key:
            label = 'loss'
        if 'MAPE' in key:
            label += 'percentage error '
        if 'mean_absolute_percentage_error' in key:
            label += 'percentage error'
        if 'MSE' in key:
            label += 'mean squared error '
        if '_T' == key[-2:]:
            label += 'T'
        if '_mf' == key[-3:]:
            label += 'mf'
        if '_p' == key[-2:]:
            label += 'p'
        if '_Tend' == key[-5:]:
            label += 'Tend'
        if label != '':
            if 'val' in key:
                ax_val.plot(epochs, values, label=label)
            else:
                ax_train.plot(epochs, values, label=label)

    ax_train.set_xlabel('epochs')
    ax_train.set_ylabel('loss')
    ax_train.set_title('training data')
    fig_train.legend()

    ax_val.set_xlabel('epochs')
    ax_val.set_ylabel('loss')
    ax_val.set_title('validation data')
    fig_val.legend()
