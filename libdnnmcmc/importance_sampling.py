'''
author: @Andreas Bott
'''


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
tfd = tfp.distributions


def weight_samples(states, measurement_indices, measurement_noise, measurement_values, plot=False):
    # prob. of each sample given the prior it is sampled from

    # calculate the probability of the state given the measurement
    measurement_dist = tfd.MultivariateNormalDiag(loc=tf.squeeze(measurement_values), scale_diag=measurement_noise)
    states_m_vals = tf.gather(states, measurement_indices, axis=1)
    sample_p_prob = measurement_dist.prob(states_m_vals)
    weights = sample_p_prob

    if plot:
        # x = np.arange(-10, 6, 1)
        # xi = 10.**x
        xi = np.arange(0.01, 1.5, 0.01)
        yi = np.array([np.shape(np.where(weights > xi[i]))[1] for i in range(len(xi))]) / np.shape(weights)[0] * 100
        fig, ax = plt.subplots()
        ax.plot(xi, yi)
        # ax.set_xscale('log')
        ax.set_xlabel('weights larger than')
        ax.set_ylabel('percentage of samples')

        weights_sort = np.sort(weights)
        xi = weights_sort.cumsum()
        yi = np.linspace(0, 100, num=len(weights))
        fig, ax = plt.subplots()
        ax.plot(xi, yi)
        ax.set_xlabel('sum of all weights')
        ax.set_ylabel('percentage of samples')

    return weights


class ImportanceSamplingResampler():
    def __init__(self, states_prior, measurement_indices, measurement_stds, measurement_values=None):
        self.states_prior = states_prior
        self.measurement_indices = measurement_indices
        self.measurement_stds = measurement_stds
        self.plot_weights = False
        if measurement_values is not None:
            self.set_measurement(measurement_values)

    def set_measurement(self, measurement_values):
        self.measurement_values = measurement_values
        weights = weight_samples(self.states_prior, self.measurement_indices, self.measurement_stds,
                                      measurement_values, plot=self.plot_weights)
        self.weights = weights/np.sum(weights)

    def resample(self, n_samples):
        if not hasattr(self, 'weights'):
            print(f'Warning: no measurement set for Importance Sampling Resampling')
            sample_ind = np.random.choice(len(self.states_prior), size=n_samples)
        else:
            sample_ind = np.random.choice(len(self.weights), size=n_samples, p=self.weights)
            print(tf.reduce_sum(tf.gather(self.weights, sample_ind)))
        return tf.gather(self.states_prior, sample_ind)
