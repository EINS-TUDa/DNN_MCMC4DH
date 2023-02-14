import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import pickle
import json
import networkx as nx
from libdnnmcmc.steady_state_solvers import SE_solver
from libdnnmcmc.state_equations import state_equations

solve_SE = SE_solver()
tfd = tfp.distributions


class ZeroTruncatedMultivariateNormal(tfd.MultivariateNormalTriL):
    '''
    adaptation of the MultivariateNormal distribution provided by tfd
    Cut off all values below zero; overwrite functions: sample, prob, log_prob accordingly
    '''
    def __init__(self, loc, scale_tril, validate_args=True, name='ZeroTruncatedMultivariateNormal'):
        parameters = dict(locals())
        self.signs = tf.math.sign(loc)
        self.sample_dim = loc.get_shape()[0]
        super().__init__(loc=tf.math.abs(loc), scale_tril=scale_tril, validate_args=validate_args, name=name)
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # parameters have the same properties (e.g. shapes) as in non-truncated normal distributions
        return tfd.MultivariateNormalTriL._parameter_properties(dtype, num_classes)

    def _mean(self):
        return self._parameters['loc']

    def sample(self, n_samples):
        # We override sample rather than _sample_n as the the super (tfd.MultivariateNormalTriL) overwrites
        # _call_sample_n instead of _sample_n and _sample_n is no longer called during self.sample().

        # initialise output tensor with all zeros
        init_samples = tf.zeros(shape=(n_samples, self.sample_dim), dtype=tf.float64)
        # check if any sample state is <0 (invalid sample) or =0 (not jet defined or removed due to production surplus)
        cond = lambda samples: tf.reduce_any(samples <= 0)

        # sample and fill valid rows into the output tensor
        def body(samples):
            new_samples = super(ZeroTruncatedMultivariateNormal, self).sample(n_samples)
            # replace samples that are not jet placed or which contain values below 0 with new samples
            #                  < overhat >  vvv actual cond. vvv      <    overhead to slice the right dimension    >
            samples = tf.where(tf.repeat(tf.reduce_all(samples > 0, axis=1, keepdims=True), self.sample_dim, axis=1),
                               samples, new_samples)

            # remove rows, which do not add up to values above zero (i.e. total demand is lower than production)
            samples = tf.where(tf.repeat(tf.reduce_sum(self.signs * samples, axis=1, keepdims=True) > 0,
                                         self.sample_dim, axis=1), samples, tf.zeros_like(samples))
            return [samples]

        [samples] = tf.while_loop(cond=cond, body=body, loop_vars=[init_samples])
        return samples * self.signs

    def _prob(self, samples):
        """
            samples containing one negative entry: pdf and gradient are all zero;
            all positive samples: same pdf and gradient as non-truncated dist.
            CAUTION: pdf-values are not normalised!
        """

        ''' _nd_prob only works for tensors of rank 2 and above -> expand dims if needed '''
        return tf.cond(tf.rank(samples) == 1,
                       lambda: self._nd_prob(tf.expand_dims(self.signs * samples, axis=0)),
                       lambda: self._nd_prob(self.signs * samples))

    def _nd_prob(self, samples):
        ''' grad for zero-samples is zero in all dim. - this is mathematically correct, but not necessarily useful '''
        M = tf.map_fn(lambda d: tf.cond(tf.reduce_all(d >= 0),
                                        lambda: super(ZeroTruncatedMultivariateNormal, self)._prob(d),
                                        lambda: 0)
                      , samples)
        return M

    def log_prob(self, samples):
        '''
            all positive samples: same as log_prob for non-truncated distribution
            samples containing at least one negative entry:
            return value: -infinity
            return value gradient: zero for positive dimensions, +1 for negative dimensions
            (this definition of the gradient is mathematically wrong but useful for Hamiltonian MCMC)

            log_prob only works for samples with rank 2 an above.
            Introducing a cond. similar to _prob leads to errors during MCMC sampling.

            According to the tfp specifications, this function should define a _log_prob instead of overwriting log_prob
            However, this leads to some circular references and wrong calculations.
            This implementation seems to work best.
        '''
        return self._nd_log_prob(self.signs * samples)

    def _nd_log_prob(self, samples):
        ''' grad for zero-samples points towards zero axis. - this is mathematically wrong, but usually useful '''
        M = tf.map_fn(lambda d: tf.cond(tf.reduce_all(d >= 0),
                                        lambda: super(ZeroTruncatedMultivariateNormal, self)._log_prob(d),
                                        lambda: tf.reduce_sum(tf.constant([-np.inf], dtype=tf.float64) - tf.math.minimum(d, 0)))
                      , samples)
        return M


class BotchedNormalDist:
    """
    "Normal Distribution", where some dimensions have zero variance

    botched_attributes only consider nonzero entries,
    true_attributes have the inflated dimensions
    """

    def __init__(self, mean, botched_cov, mask_matrix):
        """
        :param mean:        distribution mean in inflated dimensions
        :param botched_cov: covariance matrix for entries which have a variance
        :param mask_matrix: binary mask matrix
                            mapping the botched dimensions on the true ones by right hand side multiplication
        """
        self.true_mean = mean
        self.botched_mean = tf.squeeze(tf.sparse.sparse_dense_matmul(tf.sparse.transpose(mask_matrix), mean))
        self.inv_botched_mean = self.true_mean - \
                                tf.sparse.sparse_dense_matmul(mask_matrix, tf.expand_dims(self.botched_mean, axis=-1))
        self.true_cov = tf.matmul(tf.sparse.sparse_dense_matmul(mask_matrix, botched_cov),
                                  tf.transpose(tf.sparse.to_dense(mask_matrix)))
        self.botched_cov = botched_cov
        self.mask_matrix = mask_matrix

    def sample(self, n_samples):
        """ sampling in only performend in the botched dimensions """
        botched_samples = tf.constant(np.random.multivariate_normal(self.botched_mean, self.botched_cov, n_samples))
        return tf.transpose(
            tf.sparse.sparse_dense_matmul(self.mask_matrix, tf.transpose(botched_samples)) + self.inv_botched_mean)

    def marginal(self, d):
        """ returns the 1D marginal Distribution for dimension d """
        return tfd.Normal(loc=self.true_mean[d], scale=tf.sqrt(self.true_cov[d, d]))


class VirtualMeasurements:
    """
        virtual representation of a measurement in the grid
        :param installed_measurements: measurement position as dict {meas. type: [(index, prec. in % of nominal value)]}
    """

    def __init__(self, installed_measurements, SE, cycles, verbose=False):
        self.cycles = cycles
        self.SE = SE
        self.verbose = verbose
        self.installed_measure = installed_measurements
        measurement_indices = []
        measurement_noise = []
        for type in installed_measurements.keys():
            for ind in installed_measurements[type]:
                try:
                    jnd = SE.find_node[ind[0]]
                except KeyError:
                    jnd = SE.find_edge[ind[0]]
                if type == 'T':
                    offset = 0
                    val = SE.T[jnd]
                elif type == 'mf':
                    offset = SE.n_nodes
                    val = SE.mf[jnd]
                elif type == 'p':
                    offset = SE.n_nodes + SE.n_edges
                    val = SE.p[jnd]
                elif type == 'T_end':
                    offset = SE.n_nodes + SE.n_edges + SE.n_nodes
                    val = SE.T_end[jnd]
                else:
                    raise Exception('misspecified measurement type')
                measurement_indices.append(jnd + offset)
                # measurement_noise.append(tf.cast(ind[1], tf.float64))
                measurement_noise.append(tf.squeeze(ind[1] / 100 * val))
        self.measurement_indices = measurement_indices
        self.measurement_noise = measurement_noise
        self.measurement_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=(1), dtype=tf.float64),
                                                           scale_diag=measurement_noise)

    def generate_random_measurements(self, d_distribution):
        sample_demand = d_distribution.sample(1)
        state = self._get_state_from_demand(tf.squeeze(sample_demand))

        m_vals_true = tf.gather_nd(state, indices=np.expand_dims(self.measurement_indices, axis=-1))
        m_noise = self.measurement_dist.sample(1)
        return m_vals_true + tf.transpose(m_noise)

    def _get_state_from_demand(self, demand):
        SE = self.SE
        for i in range(demand.shape[0]):
            SE.Q_heat[i].assign(demand[i])
        SE.load_save_state()
        solve_SE(SE, self.cycles, verbose=self.verbose)
        return tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)


def save_measurements_to_pkl(measurement, file='temp/measurement.pkl'):
    pickle.dump(measurement, open(file, 'wb'))


def load_measurement_from_pkl(file='temp/measurement.pkl'):
    return pickle.load(open(file, 'rb'))

class LogProbDemand:
    def __init__(self, demand_prior, model, measurement_index, measurement_values, measurement_std):
        """
        This class is used to evaluate each sample during the MCMC algorithm.
        __call__ returns the un-normalised log_prob for the posterior distribution over the demands.

        :param demand_prior:           tfp.distribution for prior demand, has to have a .logprob function
        :param model:                  callable mapping demand -> states
        :param measurement_index:      position of measurements
        :param measurement_values:     measured values, same order as measurement_index
        :param measurement_variance:   measurement uncertainty, same order as measurement_index
        """
        self.demand_prior = demand_prior

        # measurement distribution (gaussian, with measurement mean and measurement variance)
        if tf.shape(measurement_values).shape == 2:
            if tf.shape(measurement_values)[0] > tf.shape(measurement_values)[1]:
                self.meas_dist = tfd.MultivariateNormalDiag(tf.transpose(measurement_values), measurement_std)
            else:
                self.meas_dist = tfd.MultivariateNormalDiag(measurement_values, measurement_std)
        else:
            self.meas_dist = tfd.MultivariateNormalDiag(measurement_values, measurement_std)

        """ this function returns the measurement values corresponding to one demand """
        self.get_measurement_values = lambda demand: tf.gather(model(demand), measurement_index, axis=1)

    def __call__(self, demands):
        M = tf.map_fn(lambda d: tf.reduce_sum(
            self.meas_dist.log_prob(self.get_measurement_values(tf.expand_dims(d, axis=0))), axis=0), demands) \
            + self.demand_prior.log_prob(demands)
        return M



def import_training_data(file_name, n_samples, f_id0=1, skip=0, dtype=tf.float64):
    """
    n_samples: int
        number of samples to be loaded in
    file_name: string
        name of the files the data is read in from;
    f_id0: integer optional
        if the file is not found, add increasing integers at the end of the file name starting with f_id0
    skip: interger optional
        if not zero: skip the first n samples
    """

    def parse_line(line):
        line = line.translate({ord(']'): None})
        _, d, _, x = line.split('[')
        d = np.fromstring(d, dtype=np.float64, sep=' ')
        x = np.fromstring(x, dtype=np.float64, sep=' ')
        return d, x

    def file_length(file_name):
        with open(file_name, 'r') as f:
            for i, _ in enumerate(f):
                pass
        return i + 1

    def get_data_format(file_name):
        with open(file_name, 'r') as f:
            d, x = parse_line(f.readline())
            return d.shape, x.shape

    def read_single_file(file_name, ds, xs, start_index, end_index, skip):
        ind = start_index
        with open(file_name, 'r') as f:
            for i, l in enumerate(f):
                if i < skip:
                    continue
                else:
                    ds[ind, :], xs[ind, :] = parse_line(l)
                    ind += 1
                    if ind == end_index:
                        break
        return ind

    try:
        data_format = get_data_format(file_name)
    except FileNotFoundError:
        data_format = get_data_format(f'{file_name}{f_id0}.csv')
    ds = np.zeros((n_samples, data_format[0][0]))
    xs = np.zeros((n_samples, data_format[1][0]))
    try:
        _ = read_single_file(file_name, ds, xs, start_index=0, end_index=n_samples, skip=skip)
    except FileNotFoundError:
        f_id = f_id0
        n_read = 0
        start_index = 0
        while start_index != n_samples:
            f = f'{file_name}{f_id}.csv'
            f_id += 1

            # skip files entirely if the number of lines is smaller than the skip value
            if skip > 0:
                f_len = file_length(f)
                if f_len <= skip:
                    # reduce skip by the number of skipped samples and continue with next file
                    skip -= f_len
                    continue

            # read single file returns last index, used as start_index in next iteration
            try:
                start_index = read_single_file(f, ds, xs, start_index, n_samples, skip)
            except FileNotFoundError:
                raise Exception(f'Number of training samples in files {file_name}{f_id0} to {file_name}{f_id - 1}'
                                f'is less than the requested number of samples ({start_index}, {n_samples})')
            skip = 0  # only skip lines in the first file read

    return [tf.constant(ds, dtype=dtype), tf.constant(xs, dtype=dtype)]

def load_scenario(scenario_name):
    """
    this function does all the setup fir a State Equation and Virtual Measurement Object and the prior distributions

    :param scenario_name:  specification, which scenario should be loaded - to add a new scenario, build a
                           specific settings file and add it to the list
    :return: SE (state equations), VM (virtual Measurements) prior (prior demand distribution)
             data_file (file path to training / validation data) n_data_flies: number of data files to be loaded
    """

    if scenario_name == 'loop':
        from data_files.inputs_loop.grid_loop import grid_file, demands, heatings, fix_dp, data_file, Ta, d_prior_mean, \
                                            d_prior_cov, measurements
    elif scenario_name == 'tree':
        from data_files.inputs_tree.grid_tree import grid_file, demands, heatings, fix_dp, data_file, Ta, d_prior_mean, \
                                            d_prior_cov, measurements

    else:
        raise Exception('Unknown scenario specification - to add a new scenario pls add an entry to'
                        'utility.load_scenario with the location of the settings')
    """
    expected imports:
        grid_file: JSON containing the gird parameter
        demands: dict {edg_index : {'Power': float/int, 'Temperature': float/int}, ... }  consumer or producer 
        heatings: dict {edg_index : {'Power': float/int, 'Temperature': float/int}, ... } slack producer
        fic_dp: dict {edg_index: p} points with fixed pressure
        data_file: string file path towards training / validation data pairs
        Ta: tf.constant(tf.float64)) ambient temperature of
        d_prior_mean: mean of the zero truncated normal distribution used as prior
        d_prior_cov: covariance matrix of the prior
        measurements: dict {'mc': [('edg_index', prec. in %), ... ], 'T': [...], 'p': [...], 'T_end': [...]'} 
    """

    def loadGridModel(jsonPaths):
        """
        Loads district heating network data from json-file to  a MeFlexWärme GridModel
        @author: Friedrich
        """
        # load json-file. Note: Path must be given as raw-string: r"<Path>"
        nd = json.load(open(jsonPaths, "r"))
        nd = [el for el in nd if 'type' in el.keys()]  # Filter for elements that have the field 'type'
        nodes = [(nd[i]['index'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Node']
        edges = [(nd[i]['from'], nd[i]['to'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Edge']

        grid = nx.DiGraph()
        grid.add_nodes_from(nodes)
        grid.add_edges_from(edges)
        return grid

    grid = loadGridModel(grid_file)

    # identify all cycles in the passive parts of the grid (looping pipes)
    sup_graph = nx.DiGraph(grid.subgraph((node for node, data in grid.nodes(data=True) if data['nw_section'] == 'Sup')))
    ret_graph = nx.DiGraph(grid.subgraph((node for node, data in grid.nodes(data=True) if data['nw_section'] == 'Ret')))
    
    def find_all_cycles(graph_input):
        g = nx.DiGraph(graph_input)
        cycles = []
        while True:
            try:
                cycle = nx.algorithms.cycles.find_cycle(g, orientation='ignore')
                cycles.append(cycle)
            except nx.exception.NetworkXNoCycle:  # exception appears if no cycle is found
                break
            for u, v, direction in cycle:
                g.remove_edge(u, v)
        return cycles
    
    cycles = []
    cycles.extend(find_all_cycles(sup_graph))
    cycles.extend(find_all_cycles(ret_graph))
    if not cycles == []:
        cycles = [[(grid.edges[(e[0], e[1])]['index'], e[2]) for e in c] for c in cycles]

    # pipe parametrisations
    pipes_db = pd.read_excel('data_files/Library_Pipe_Parameters.xlsx', index_col=0, engine='openpyxl')

    # store nodes and edges in list format for easier access in the SE object:
    nodes = []
    edges = []

    for node_idx in grid.nodes():
        nodes.append(grid.nodes[node_idx])

    for edge_idx in grid.edges():
        edge_vals = grid.edges[edge_idx]
        if edge_vals['edge control type'] == 'passive':
            edge_vals['temp_loss_coeff'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'Norm-Wärme-übergangskoeffizent']
            edge_vals['diameter'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'hydraulischer Durchmesser']
            if edge_vals['nw_section'] == 'sup':
                edge_vals['fd_nom'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'fd_nom 110C']
            else:
                edge_vals['fd_nom'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'fd_nom 65C']
            edge_vals['bend_factor'] = edge_vals.get('pressure loss factor[-]', 1)
        # edges contains all active and passive edges
        edges.append(edge_vals)

    # construct state equations object
    SE = state_equations(edges=edges, nodes=nodes, demands=demands, heatings=heatings, fix_dp=fix_dp, Ta=Ta)
    SE.set_init_state()

    # solve for mean conditions and save resulting state
    solve_SE = SE_solver()
    solve_SE(SE, cycles)
    SE.set_save_state()

    ''' setup demand distribution '''
    d_prior_dist = ZeroTruncatedMultivariateNormal(loc=d_prior_mean, scale_tril=tf.linalg.cholesky(d_prior_cov),
                                                   validate_args=True, name='prior_demand_distribution')

    ''' initialise virtual measurements object '''
    VM = VirtualMeasurements(measurements, SE, cycles)

    return SE, VM, d_prior_dist, data_file
