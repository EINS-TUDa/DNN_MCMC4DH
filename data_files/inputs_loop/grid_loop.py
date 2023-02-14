import tensorflow as tf

'''
this file contains the settings for the state estimation, e.g. prior demands and measurement positions
'''
# grid and data files:
grid_file = 'data_files/inputs_loop/grid_loop.json'
data_file = 'data_files/results_loop/MC_results_pt_'

# prior demand distribution: MVN
d_prior_mean = tf.constant([200, 200, 20, 200], dtype=tf.float64)

d_prior_cov = tf.constant( [[7000,   0,   0, -6300],
                            [  0, 4000,   0,     0],
                            [  0,    0, 100,     0],
                            [ -6300, 0,   0,  7000]], dtype=tf.float64)

heatings = {'heating': {'Power': -620, 'Temperature': 120}}
demands  = {'Dem_B': {'Power': d_prior_mean[0], 'Temperature': 50},
            'Dem_E': {'Power': d_prior_mean[1], 'Temperature': 55},
            'Dem_F': {'Power': d_prior_mean[2], 'Temperature': 60},
            'Dem_I': {'Power': d_prior_mean[3], 'Temperature': 40}}

# measurement position and percentage measurement error
measurements = {'mf': [('ADS', 1)], 'T': [('AR', 1)], 'p': [], 'T_end': []}

# points of constant pressure (usually next to the powerplant)
fix_dp = {'AR': 3, 'AS': 6.5}

# ambient temperature
Ta = tf.constant(10., dtype=tf.float64)
