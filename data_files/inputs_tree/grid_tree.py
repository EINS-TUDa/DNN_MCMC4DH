import tensorflow as tf
import pandas as pd

'''
this file contains the settings for the state estimation, e.g. prior demands and measurement positions
'''
# grid and data files:
grid_file = 'data_files/inputs_tree/grid_tree.json'
data_file = 'data_files/results_tree/MC_results_r_'


#%% prior demand distribution: MVN
temp_file = 'data_files/inputs_tree/temperatures.csv'
temp_pd = pd.read_csv(temp_file)
mean_file = 'data_files/inputs_tree/mean_values_demand.csv'
mean_pd = pd.read_csv(mean_file)
cov_file = 'data_files/inputs_tree/cov_structure_demand.csv'
cov_pd = pd.read_csv(cov_file)
cov_pd.set_index('Unnamed: 0', inplace=True, drop=True)


order = list(mean_pd.columns)[1:]
# change order of entries to match the order in the training data set:
right_order = [order[0], *order[2:-1], order[1], order[-1]]

d_prior_mean = tf.constant(mean_pd.loc[mean_pd.index[0], right_order].values, dtype=tf.float64)
d_prior_cov = tf.constant(cov_pd.loc[right_order, right_order].values, dtype=tf.float64)


#%%
heatings = {'heating': {'Power': -400, 'Temperature': 120}}
demands = dict()
for d in order:
    demands[d] = {'Power': mean_pd.loc[:, d].values[0], 'Temperature': temp_pd.loc[:, d].values[0]}

# measurement position and percentage measurement error
measurements = {'mf': [('edge_13', 1)], 'T': [('DA1063', 1)], 'p': [], 'T_end': []}   # power plant massflow and return temperature

# points of constant pressure (usually next to the powerplant)
fix_dp = {'DA1063': 3.5, 'DA1062': 6.5}

# ambient temperature
Ta = tf.constant(10., dtype=tf.float64)
