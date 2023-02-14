'''
@author: Andreas Bott, Tim Janke
'''


import numpy as np
import tensorflow as tf


class consistency_loss():
    """
    This function evaluates the violation of the state equations sorted by equation type
    """
    def __init__(self, SE):
        self.n_nodes = SE.n_nodes
        self.n_edges = SE.n_edges
        self.SE = SE
        self.n_mf_conservation_nodes = SE.n_nodes - 1
        self.n_T_mix_nodes = np.sum(SE.B_mask > 0)

        ind_tend = []
        ind_p = []
        ind_Q = []
        i = 0
        for e in SE.edges:
            if e['edge control type'] == 'passive':
                ind_tend.append(i)
                i += 1
                ind_p.append(i)
                i += 1
            elif e['index'] in SE.demands.keys():
                ind_Q.append(i)
                i += 1
            else:
                i += 1

        self.ind_tend = ind_tend
        self.ind_p = ind_p
        self.ind_Q = ind_Q

    def __call__(self, state_samples, demand_samples):
        [T, mf, p, T_end] = tf.split(state_samples, num_or_size_splits=[self.n_nodes, self.n_edges, self.n_nodes, self.n_edges],
                                     axis=-1)
        loss = tf.map_fn(lambda inp: self.SE.evaluate_state_equations('forwardpass', T=tf.expand_dims(inp[0], axis=-1),
                                                               mf=tf.expand_dims(inp[1], axis=-1),
                                                               p=tf.expand_dims(inp[2], axis=-1),
                                                               T_end=tf.expand_dims(inp[3], axis=-1),
                                                               Q_heat=tf.expand_dims(inp[4], axis=-1)),
                  (T, mf, p, T_end, demand_samples), dtype=tf.float64)
        loss = tf.squeeze(loss)

        loss_mfconv = loss.numpy()[0:self.n_mf_conservation_nodes]
        loss_tnode = loss.numpy()[self.n_mf_conservation_nodes:self.n_mf_conservation_nodes+self.n_T_mix_nodes]
        loss_tend = tf.gather(loss, self.ind_tend)
        loss_p = tf.gather(loss, self.ind_p)
        loss_Q = tf.gather(loss, self.ind_Q)
        return loss_mfconv, loss_tnode, loss_tend, loss_p, loss_Q, loss


def beauty_report_qdists(qdist, SE, mode='absolute'):
    """
    qdist: (n x m) np.array containing the quantile-distances
            n: different MCMC runs
            m: state dimensions

    prints the mean and maximum deviations for each state dimension averaged over all MCMC runs
    """
    n_nodes = SE.n_nodes
    n_edges = SE.n_edges
    m = SE.masks
    if mode == 'absolute':
        str_end = ''
    else:
        str_end = '%'
    if np.shape(qdist)[1] == 2*n_nodes + 2*n_edges:
        qd = np.abs(qdist[:, 0:n_nodes][:, np.where(m[0])[0]])
        print(f'MAE dist T {np.mean(np.mean(np.abs(qd))):.2f} {str_end}')
        print(f'Max dist T {np.mean(np.max(qd, axis=1)):.2f} {str_end}')
        qd = np.abs(qdist[:, n_nodes:n_nodes+n_edges][:, np.where(m[1])[0]])
        print(f'MAE dist mf {np.mean(np.abs(qd)):.2f} {str_end}')
        print(f'Max dist mf {np.mean(np.max(np.abs(qd), axis=1)):.2f} {str_end}')
        qd = qdist[:, n_nodes+n_edges:2*n_nodes+n_edges][:, np.where(m[2])[0]]
        if mode == 'absolute':
            print(f'MAE dist p {np.mean(np.abs(qd))*1.e3:.2f}')
            print(f'Max dist p {np.mean(np.max(np.abs(qd), axis=1)) * 1e3:.2f}')
        else:
            print(f'MAE dist p {np.mean(np.abs(qd)):.2f} {str_end}')
            print(f'Max dist p {np.mean(np.max(np.abs(qd), axis=1)):.2f} {str_end}')
        qd = qdist[:, 2*n_nodes+n_edges:2*n_nodes+2*n_edges][:, np.where(m[3])[0]]
        print(f'MAE dist T_end {np.mean(np.abs(qd)):.2f} {str_end}')
        print(f'Max dist T_end {np.mean(np.max(np.abs(qd), axis=1)):.2f} {str_end}')
    else:
        a = 0
        b = int(np.sum(m[0]))
        qd = qdist[a:a+b]
        print(f'MAE dist T {np.mean(np.abs(qd)):.2f}')
        print(f'Max dist T {np.max(np.abs(qd)):.2f}')
        a += b
        b = int(np.sum(m[1]))
        qd = qdist[a:a+b]
        print(f'MAE dist mf {np.mean(np.abs(qd)):.2f}')
        print(f'Max dist mf {np.max(np.abs(qd)):.2f}')
        a += b
        b = int(np.sum(m[2]))
        qd = qdist[a:a+b]
        print(f'MAE dist p {np.mean(np.abs(qd))*1e3:.2f}')
        print(f'Max dist p {np.max(np.abs(qd))*1e3:.2f}')
        a += b
        b = int(np.sum(m[3]))
        qd = qdist[a:a+b]
        print(f'MAE dist T_end {np.mean(np.abs(qd)):.2f}')
        print(f'Max dist T_end {np.max(np.abs(qd)):.2f}')

        print('controll value: a = ', a, ' b = ', b, 'a + b = ', a+b)



############### define loss ################
@tf.function()
def ed(y_data, y_model, data_weights=None, model_weights=None, epsilon=0):
    """
    @Tim Janke
    Compute Energy distance Args:
    y_data (tf.tensor, shape 1xDxN): Samples from true distribution.
    y_model (tf. tensor, shape 1xDxM): Samples from model.
    Returns:
    tf.float: Energy distance for batch


    thoughts on implementation:
        - matmul: returns a weighted mean, true mean if all weights are equal
        - partial vectorisation: tradeoff between required storage space and computational costs.
        - tf.map_fn and tf.vectorized_map: sequential and vectorised mapping for functions.
        -> later: equal in performance and space requirements to implementation with vector multiplication
        -> former: way less storage, but at a much higher computational cost
        fully vectorised version requires a lot of ram for large number of vectors which can not be provided.
        non-vectorised version is much slower -> vectorised only outer multiplication
    """

    ''' partially vectorised Version: '''
    n_samples_model = tf.cast(tf.shape(y_model)[0], dtype=tf.float64)
    n_samples_data = tf.cast(tf.shape(y_data)[0], dtype=tf.float64)
    N = y_model.shape[0]
    M = y_data.shape[0]

    if data_weights is None:
        data_weights = tf.ones(tf.shape(y_data)[0], dtype=tf.float64) / n_samples_data
    else:
        data_weights = data_weights / tf.reduce_sum(data_weights)  # normalise weights
    if model_weights is None:
        model_weights = tf.ones(tf.shape(y_model)[0], dtype=tf.float64) / n_samples_model
    else:
        model_weights = model_weights / tf.reduce_sum(model_weights)  # normalise weights

    # expand weights to be an 1xN / 1xM vector
    data_weights = tf.expand_dims(data_weights, axis=0)
    model_weights = tf.expand_dims(model_weights, axis=0)

    # vector-distance:
    @tf.function()
    def v_dist(a, b):
        return tf.sqrt(tf.reduce_sum(tf.square(a - b)))

    mmd_11 = tf.matmul(data_weights, tf.squeeze(tf.vectorized_map(
        lambda y_0: tf.matmul(tf.expand_dims(tf.map_fn(lambda y: v_dist(y, y_0), y_data), axis=0), data_weights,
                              transpose_b=True), y_data), axis=-1))
    mmd_22 = tf.matmul(model_weights, tf.squeeze(tf.vectorized_map(
        lambda y_0: tf.matmul(tf.expand_dims(tf.map_fn(lambda y: v_dist(y, y_0), y_model), axis=0), model_weights,
                              transpose_b=True), y_model), axis=-1))
    mmd_12 = tf.matmul(data_weights, tf.squeeze(tf.vectorized_map(
        lambda y_0: tf.matmul(tf.expand_dims(tf.map_fn(lambda y: v_dist(y, y_0), y_model), axis=0), model_weights,
                              transpose_b=True), y_data), axis=-1))

    loss = 2 * mmd_12 - mmd_22 - mmd_11
    return tf.squeeze(loss)


def segmented_ed(y_data, y_model, segments=[]):
    """
    calculates the Energy Distance individually for segments of the results vector.
    """
    eds = tf.Variable(tf.zeros((len(segments),1), dtype=tf.float64))
    for i, (lb, ub) in enumerate(segments):
        eds[i].assign([ed(y_data[:, lb:ub], y_model[:, lb:ub])])
    return tf.squeeze(eds)
