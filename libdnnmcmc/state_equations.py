# -*- coding: utf-8 -*-
"""
This file defines the state_equation object.
The purpose of this object is to summarize all state variables and steady state equations; additionally it also defines
functions subroutines used to solve the steady-state system. Refer to ./steady_state_solvers.py for more details on how
these subroutines are integrated is the solving algorithm.

@author: Andreas Bott
"""

import tensorflow as tf
import numpy as np
import networkx as nx
import warnings

from typing import Tuple, Set, List
tf.keras.backend.set_floatx('float64')

# custom exception, raised if solving the temperatures for given mass flows does not converge
# this can happen due to initially bad values for mass flows or misdefined grid topologies
class TemperatureProbagationException(Exception):
    def __init__(self, message = 'Temperature propagation does not reach all nodes'):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class state_equations():
    def __init__(self, edges, nodes, demands, heatings, fix_dp: dict, use_demands=True,
                 Ta=tf.constant(5., dtype=tf.float64)):
        """
        :param edges:       list of all edges in the grid
        :param nodes:       list of all nodes in the grid
        :param demands:     list of all demands in the grid
        :param heatings:    list of all heatings in the grid (only slack heatings, fixed heatings are considered to be negative demands)
        :param fix_dp:      dict of nodes, where the pressure is fixed; key: node-index, val: pressure value
        :param use_demands: bool; default True;
                            if true: consider demand coverage when calculating the missmatch of the SE equations
        :param Ta:          ambiant temperature
        """
        self.use_demands = use_demands
        # nodes and edges:
        self.edges = edges
        self.nodes = nodes
        self.n_edges = len(edges)
        self.n_nodes = len(nodes)
        self.n_demands = len(demands)
        self.n_active_edges = len([edg for edg in edges if edg['edge control type'] == 'active'])
        find_node = dict()  # dict to find the numerical index of a node by name
        find_edge = dict()  # dict to find the numerical index of an edge by name
        for i in range(self.n_nodes):
            find_node[nodes[i]['index']] = i
        for i in range(self.n_edges):
            find_edge[edges[i]['index']] = i
        self.find_node = find_node
        self.find_edge = find_edge

        # demands and heatings:
        self.demands = demands
        self.heatings = heatings
        self.fix_dp = fix_dp

        # state variables
        self.T = tf.Variable(tf.ones((self.n_nodes, 1), dtype=tf.float64), dtype=tf.float64, name='temperaturs')
        self.p = tf.Variable(tf.ones((self.n_nodes, 1), dtype=tf.float64), dtype=tf.float64, name='pressures')
        self.mf = tf.Variable(tf.ones((self.n_edges, 1), dtype=tf.float64), dtype=tf.float64, name='mass_flows')
        self.T_end = tf.Variable(tf.ones((self.n_edges, 1), dtype=tf.float64), dtype=tf.float64,
                                 name='end-of-line_temperature')
        self.Q_heat = tf.Variable(tf.ones(self.n_active_edges - 1, dtype=tf.float64), dtype=tf.float64,
                                  name='demand_power_values')
        self.Q_T_end = tf.Variable(tf.ones(self.n_active_edges - 1, dtype=tf.float64), dtype=tf.float64,
                                   name='feed_in_temperatures')

        # constant parameter and physical properties
        self.Ta = Ta
        self.cp = tf.constant(4.180, name='cp_water_in_kJ/kgK', dtype=tf.float64)
        self.pi = tf.constant(np.pi, name='pi', dtype=tf.float64)
        self.rho = tf.constant(997., name='density_water', dtype=tf.float64)

        # setting up connectivity matrix A, masks and setting initial values for state variables

        '''
        setting up connectivity matrix A encoding the graph structure of the grid. 
        
        setting up masks and mask matrices:  
            masks: boolean masks which are the same size as the corresponding state variable dimension 
                   -> mask=0 if the state variable is a priori fixed and 1 otherwise
            mask-matrix: boolean matrix M which maps a vector V_red containing only not fixed states to the dimensions 
                         of a state vector V_full containing all states; V_full = M@V_red 
                         Fixed states are filled with zeros.
        '''
        T_mask = np.ones((self.n_nodes, 1))
        p_mask = np.ones((self.n_nodes, 1))
        Tend_mask = np.ones((self.n_edges, 1))
        mf_mask = np.ones((self.n_edges, 1))
        A_row = []  # setup for sparse connectivity matrix A
        A_col = []
        A_val = []
        A_active_row = []  # identity mapping matrix active mass-flows
        A_active_col = []
        A_active_val = []
        dem_ind = dict()
        d_count = 0

        for i in range(self.n_nodes):
            nd = self.nodes[i]
            if nd['index'] in self.fix_dp.keys():
                p_mask[i] = 0
                self.p[i].assign([fix_dp[nd['index']]])

        for i, edg in enumerate(self.edges):
            n_from = self.find_node[edg['from']]
            n_to = self.find_node[edg['to']]
            # add entry in A
            A_row.extend([n_from, n_to])
            A_col.extend([i, i])
            A_val.extend([-1., 1.])

            if edg['edge control type'] == 'active':
                T_mask[n_to] = 0  # the temperature at the end of active edge is fixed.
                Tend_mask[i] = 0
                # for now: heating and demand
                if edg['index'] in self.demands.keys():
                    dem_ind[edg['index']] = d_count
                    self.Q_heat[d_count].assign(self.demands[edg['index']]['Power'])
                    self.Q_T_end[d_count].assign(self.demands[edg['index']]['Temperature'])
                    A_active_row.append(d_count)
                    A_active_col.append(i)
                    A_active_val.append(1)
                    d_count += 1

        # construct connectivity matrix A
        A = tf.SparseTensor(indices=list(zip(A_row, A_col)), values=tf.constant(A_val, dtype=tf.float64),
                            dense_shape=[self.n_nodes, self.n_edges])
        A = tf.sparse.reorder(A)
        A_red_size = tf.cast((tf.shape(A) - [1, 0]), tf.int64)

        A_active = tf.SparseTensor(indices=list(zip(A_active_row, A_active_col)),
                                   values=tf.constant(A_active_val, dtype=tf.float64),
                                   dense_shape=[d_count, self.n_edges])

        self.masks = [T_mask, mf_mask, p_mask, Tend_mask]
        self.state_dimension_segments = [
            (int(sum(np.sum(self.masks[i]) for i in range(j))), int(sum(np.sum(self.masks[i]) for i in range(j + 1))))
            for j in range(len(self.masks))]
        self.A = A
        self.A_active = A_active
        # B_mask: temperature mixing equations for which all inflows have a fixed end of line temperature
        self.B_mask = tf.squeeze(1 - tf.sparse.sparse_dense_matmul(A, tf.cast((1 - Tend_mask), tf.float64)))
        self.A_red_size = A_red_size
        self.dem_ind = dem_ind
        self.mask_matrix = [self.get_mask_matrix(self.masks[i]) for i in range(len(self.masks))]
        self.mask_matrix_full = self.combine_mask_matrix(self.mask_matrix)
        self.set_init_state()

    def get_mask_matrix(self, mask):
        """ transforms a mask (list) to a mask-matrix  """
        rows = np.nonzero(mask)[0].tolist()
        n = int(np.sum(mask))  # number on not-masked entries
        cols = list(range(n))
        ones = tf.ones(n, dtype=tf.float64)
        mask_matrix = tf.SparseTensor(indices=list(zip(rows, cols)), values=ones, dense_shape=[len(mask), n])
        mask_matrix = tf.sparse.reorder(mask_matrix)
        return mask_matrix

    def combine_mask_matrix(self, masks):
        M_ind = masks[0].indices
        M_val = masks[0].values
        shape = masks[0].shape
        for i in range(1, len(masks)):
            M_ind = tf.concat([M_ind, masks[i].indices + tf.cast(shape, tf.int64)], axis=0)
            M_val = tf.concat([M_val, masks[i].values], axis=0)
            shape = tf.add(shape, masks[i].shape)
        M = tf.SparseTensor(indices=M_ind, values=M_val, dense_shape=tf.cast(shape, tf.int64))
        return tf.sparse.reorder(M)

    def set_init_state(self, t_ret=40):
        # sets all state variables to initial values
        for i, edg in enumerate(self.edges):
            n_from = self.find_node[edg['from']]
            n_to = self.find_node[edg['to']]

            if edg['edge control type'] == 'active':
                # for now: heating and demand
                if edg['index'] in self.heatings.keys():
                    self.T[n_to].assign([self.heatings[edg['index']]['Temperature']])
                    self.T_end[i].assign([self.heatings[edg['index']]['Temperature']])
                    self.H_T_end = tf.Variable([self.heatings[edg['index']]['Temperature']], dtype=tf.float64)
                    # Q_heat.assign([heatings[edg['index']]['Power']])
                else:
                    self.T[n_to].assign([self.demands[edg['index']]['Temperature']])
                    self.T_end[i].assign([self.demands[edg['index']]['Temperature']])
            else:
                # use masks to avoid replacing fixed temperatures determined by active edges
                if edg['nw_section'] == 'Sup':
                    if self.masks[0][n_to]:
                        self.T[n_to].assign([self.heatings[list(self.heatings.keys())[0]]['Temperature']])
                    if self.masks[0][n_from]:
                        self.T[n_from].assign([self.heatings[list(self.heatings.keys())[0]]['Temperature']])
                    if self.masks[3][i]:
                        self.T_end[i].assign([self.heatings[list(self.heatings.keys())[0]]['Temperature']])
                else:
                    if self.masks[0][n_to]:
                        self.T[n_to].assign([tf.cast(t_ret, tf.float64)])
                    if self.masks[0][n_from]:
                        self.T[n_from].assign([tf.cast(t_ret, tf.float64)])
                    if self.masks[3][i]:
                        self.T_end[i].assign([tf.cast(t_ret, tf.float64)])
        self.init_state = [tf.identity(self.T), tf.identity(self.mf), tf.identity(self.p), tf.identity(self.T_end)]

    def set_save_state(self):
        # save a 'hardcopy' of the initial state
        self.save_state = [tf.identity(self.T), tf.identity(self.mf), tf.identity(self.p), tf.identity(self.T_end),
                           tf.identity(self.Q_heat)]

    def load_init_state(self):
        self.T.assign(self.init_state[0])
        self.mf.assign(self.init_state[1])
        self.p.assign(self.init_state[2])
        self.T_end.assign(self.init_state[3])

    def load_save_state(self):
        # resets the state of all state variables to its saved values - if no state is saved, set to init state
        if hasattr(self, 'save_state'):
            self.T.assign(self.save_state[0])
            self.mf.assign(self.save_state[1])
            self.p.assign(self.save_state[2])
            self.T_end.assign(self.save_state[3])
            self.Q_heat.assign(self.save_state[4])
        else:
            self.load_init_state()

    # edge functions
    @tf.function
    def eq_pipe_T(self, mf, T0, l, Lmbda):
        return (T0 - self.Ta) * tf.math.exp(
            tf.math.divide_no_nan(tf.cast(-l * Lmbda, tf.float64), (1000 * self.cp * mf))) + self.Ta

    @tf.function
    def eq_pipe_p(self, mf, fd, l, d):
        return fd * 8 * l * mf * tf.math.abs(mf) / (self.pi ** 2 * self.rho * d ** 5) * 1e-5

    @tf.function
    def eq_pipe_p_lin(self, mf0, fd, l, d):  # 1st derivertive of eq_pipe_p at mf0
        return 2 * fd * 8 * l * tf.math.abs(mf0) / (self.pi ** 2 * self.rho * d ** 5)

    @tf.function
    def eq_demand_Q(self, mf, Q, Ts, Tr):
        return mf * self.cp * (Ts - Tr) - Q

    @tf.function
    def eq_fix_dp(self, pf, pt, p_set, lambda1=1.e-2, lambda2=1.e0):
        return lambda1 * tf.clip_by_value(pt - pf - p_set, 0, 1.e7) + lambda2 * tf.clip_by_value(pt - pf - p_set, -1.e7, 0)


    @tf.function
    def evaluate_state_equations(self, mode, T=None, mf=None, p=None, T_end=None, Q_heat=None):
        if tf.executing_eagerly():
            warnings.warn('state equations evaluated eagerly!')
        '''
        evaluates the state equations,
        modes:
            forwardpass -> calculates the loss for given variables
            gradient -> returns loss and gradient of squared sum of the loss
            jacobian -> retruns loss and jacobian of the loss with respect to the state variables
            demand jacobian ->  returns jacobian of the state variables with respect to the demand inputs
        '''
        # alias state variables for readability
        A = self.A
        Q_T_end = self.Q_T_end
        H_T_end = self.H_T_end
        demands = self.demands
        heatings = self.heatings
        if Q_heat is None:
            Q_heat = self.Q_heat
        if T is None:
            T = self.T
        if mf is None:
            mf = self.mf
        if p is None:
            p = self.p
        if T_end is None:
            T_end = self.T_end
        if mode == 'demand jacobian' or tf.executing_eagerly():
            persistence = True
            # for this case two jac have to be calculated, forcing for persistent=True for the GradientTape
        else:
            persistence = False
        with tf.GradientTape(persistent=persistence) as tape:
            # loss 1: massflow conservation in nodes - one equation in A is lin. dep. on the others -> drop one row
            A_red = tf.sparse.slice(A, start=[0, 0], size=self.A_red_size)
            if mode == 'gradient':
                loss = tf.reduce_sum(tf.sparse.sparse_dense_matmul(A_red, mf) ** 2)
            else:
                loss = (tf.sparse.sparse_dense_matmul(A_red, mf))
            # temperature in each nodes:
            """
            T = Sum(m*T_end)/Sum(m)  -> B: inflow matrix, matmul leads to the sum in the numerator
            if Sum(B, axis=1) == 0 (no incoming mass flow, e.g. at zero demand) this leads to a 0/0 division
            add self.Ta*1.e-15/1.e-15 to get analytical solution Ta in this case, 1.e-15 to "ignore" it if mf != 0
            """
            B = tf.clip_by_value(
                tf.math.multiply(tf.expand_dims(tf.sparse.to_dense(A), axis=0),
                                 tf.expand_dims(tf.transpose(mf), axis=1)), 0, 1.e7)
            l = T - tf.squeeze(
                tf.transpose((tf.matmul(B, tf.expand_dims(tf.transpose(T_end), axis=-1)) + self.Ta * 1.e-15) /
                             (tf.reduce_sum(B, axis=-1, keepdims=True) + 1.e-15)), axis=0)

            l = tf.boolean_mask(l, self.B_mask, axis=0)

            if mode == 'gradient':
                loss += tf.reduce_sum(l ** 2, axis=0)
            else:
                loss = tf.concat([loss, l], axis=0)

            # equations for each edges:
            for i in range(self.n_edges):
                edg = self.edges[i]
                n_from = self.find_node[edg['from']]
                n_to = self.find_node[edg['to']]

                # edge equations:
                if edg['edge control type'] == 'passive':
                    # temperature loss among the pipe:
                    temp = tf.cond(
                        pred=mf[i, :] != 0,
                        true_fn=lambda mf=mf, n_from=n_from, n_to=n_to: (mf[i, :] + tf.abs(mf[i, :])) / (2 * mf[i, :]) * T[n_from, :] + \
                                        (mf[i, :] - tf.abs(mf[i, :])) / (2 * mf[i, :]) * T[n_to, :],
                        false_fn=lambda: self.Ta)
                    l = T_end[i, :] - self.eq_pipe_T(tf.abs(mf[i, :]), temp, edg['temp_loss_coeff'], edg['length [m]'])
                    if mode == 'gradient':
                        loss += tf.reduce_sum(l ** 2, axis=0)
                    else:
                        loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)

                    # pressure loss among the pipe:
                    l = (p[n_from] - p[n_to] - self.eq_pipe_p(mf[i], edg['fd_nom'],
                                                              edg['length [m]'] * edg['bend_factor'],
                                                              edg['diameter']))
                    if mode == 'gradient':
                        loss += tf.reduce_sum(l ** 2, axis=0)
                    else:
                        loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)

                # demand coverage:
                elif edg['index'] in demands.keys():
                    if self.use_demands:
                        l = (self.eq_demand_Q(mf[i], Q_heat[self.dem_ind[edg['index']]], T[n_from], T_end[i]))
                        if mode == 'gradient':
                            loss += tf.reduce_sum(l ** 2, axis=0)
                        else:
                            loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)

        if mode == 'forwardpass':
            return loss
        elif mode == 'gradient':
            gradients = tape.gradient(loss, [T, mf, p, T_end])
            masked_gradients = [gradients[i] * self.masks[i] for i in range(len(gradients))]
            return masked_gradients, loss
        elif mode == 'jacobian':
            jacobian = tape.jacobian(loss, [T, mf, p, T_end], experimental_use_pfor=False)
            jacobian = [tf.squeeze(jacobian[i]) for i in range(len(jacobian))]
            masked_jacobian = [tf.boolean_mask(jacobian[i], self.masks[i][:, 0], axis=1) for i in range(len(jacobian))]
            return masked_jacobian, loss
        elif mode == 'demand jacobian':
            # implicit function theorem to calculate
            #        d(T, mf p, T_end) / d(Q_heat) = - (d(loss) / d(T, mf, p, T_end)) ^-1 * (d(loss) / d(Q_heat)
            # <=> -  d(loss) / d(T, mf, p, T_end) * d(T, mf p, T_end) / d(Q_heat) = (d(loss) / d(Q_heat)
            #                                       |--      return value    ---|
            jacobian_sv = tape.jacobian(loss, [T, mf, p, T_end], experimental_use_pfor=False)
            jacobian_sv = [tf.squeeze(jacobian_sv[i]) for i in range(len(jacobian_sv))]
            jacobian_sv = [tf.boolean_mask(jacobian_sv[i], self.masks[i][:, 0], axis=1) for i in
                           range(len(jacobian_sv))]
            jacobian_sv = tf.concat(jacobian_sv, axis=1)
            jacobian_dem = tape.jacobian(loss, Q_heat, experimental_use_pfor=False)
            jacobian_dem = tf.squeeze(jacobian_dem)

            if tf.rank(jacobian_dem) == 1:
                jacobian_dem = tf.expand_dims(jacobian_dem, axis=-1)
            jac = -tf.linalg.lstsq(matrix=jacobian_sv, rhs=jacobian_dem, fast=False)
            return jac

    def set_demand_from_grid(self):
        """
        calculates the power consumption that corresponds to the actual temperatures and mass flows at the demands
        """
        for i, dem in enumerate(self.demands.keys()):
            dem_pos = self.find_edge[dem]
            Ts = self.T[self.find_node[self.edges[dem_pos]['from']]]
            dem_val = self.eq_demand_Q(self.mf[dem_pos], 0, Ts, self.T_end[dem_pos])
            self.Q_heat[self.dem_ind[dem]].assign(tf.squeeze(dem_val))
        return self.Q_heat

    def linear_cycle_equations(self, cycles):
        """
        returns the matrix and right hand side for the cycle equations A * mf = b
        """
        A_c = np.zeros(shape=(len(cycles), self.n_edges), dtype='float64')
        b_c = np.zeros(shape=(len(cycles)), dtype='float64')
        for j, c in enumerate(cycles):
            for e, direction in c:
                i = self.find_edge[e]
                edg = self.edges[i]
                # pressure_loss is given by mf * dp_mf
                # A_c @ mf -> sum up all pressure losses among the cycle
                dp_mf = self.eq_pipe_p_lin(self.mf[i], tf.constant(edg['fd_nom'], dtype=tf.float64),
                                           tf.constant(edg['length [m]'] * edg['bend_factor'], dtype=tf.float64),
                                           tf.constant(edg['diameter'], dtype=tf.float64))

                A_c[j, i] = {'forward': 1, 'reverse': -1}[direction] * dp_mf
                b = self.eq_pipe_p(self.mf[i], tf.constant(edg['fd_nom'], dtype=tf.float64),
                                   tf.constant(edg['length [m]'] * edg['bend_factor'], dtype=tf.float64),
                                   tf.constant(edg['diameter'], dtype=tf.float64))

                b_c[j] += b
        b = tf.constant(b_c, dtype=tf.float64)
        A = tf.constant(A_c, dtype=tf.float64)
        return A, b

    def solve_massflow_fixed_temp(self, mf_vals, cycles=[]):
        '''
        solves the massflow conservation equations as well as the heat demand equations for fixed tempeatures

        if cycles != 0: add condition for the pressure loss among the cycle to be zero (locally linearised)
        '''
        # alias state variables for readability
        A = self.A
        A_active = self.A_active
        demands = self.demands
        Q_heat = self.Q_heat
        T = self.T
        A_red = tf.sparse.slice(A, start=[0, 0], size=self.A_red_size)
        A_ext = tf.concat([tf.sparse.to_dense(A_red), tf.sparse.to_dense(A_active)], axis=0)
        # construct left hand side:
        bs = tf.Variable(tf.zeros(tf.shape(A_ext)[0], dtype=tf.float64))
        offset = self.A_red_size[0]
        d_count = 0
        for i, edg in enumerate(self.edges):
            # edg = self.edges[i]
            if edg['edge control type'] == 'active':
                if edg['index'] in demands.keys():
                    if mf_vals is None:
                        mf_dem = Q_heat[self.dem_ind[edg['index']]] / \
                                 (self.cp * (T[self.find_node[edg['from']]] - demands[edg['index']]['Temperature']))
                        if tf.math.is_nan(mf_dem):
                            # mf is set to none, if the demand is zero and the temperature at both sides reaches Ta
                            # (division 0/0)
                            mf_dem = tf.constant(0, dtype=tf.float64)
                        if mf_dem < 0:  # demand mass flows should never be less than zero during internal calculations.
                            # For prosumer, the edge orientation gets switched, denoted by the key-entries in the
                            # steady-state-simulator.
                            mf_dem = tf.constant([1], dtype=tf.float64)
                            # set mf_dem = 1 to prevent everything from blowing up, 1 should be in a reasonable range
                    else:   # if mfs are passed:
                        mf_dem = mf_vals[self.dem_ind[edg['index']]]
                    bs[offset + d_count].assign(tf.squeeze(mf_dem))
                    d_count += 1
        # if needed: expand the calculation to incorporate cycles:
        if not cycles == []:
            A_cycles, bs_cycles = self.linear_cycle_equations(cycles)
            A_ext = tf.concat([A_ext, A_cycles], axis=0)
            bs = tf.concat([bs, bs_cycles], axis=0)

        # solve linear system of equations: - note: lstsq is faster in general,
        # however if a demand is 0 we want a true zero mass flow for the corresponding edges. The numerical precision
        # of tf.lstsq is not sufficient in these cases
        mf_res = tf.linalg.solve(matrix=A_ext, rhs=tf.expand_dims(bs, axis=-1))
        return mf_res

    @tf.function
    def evaluate_mf_equations(self, mode, dem_mf_vals, cycles=[]):
        """
        calculates the loss due to mass-flow missmatches and for pressure losses in cycles if there are given any
        does not assign values - acts more like evaluate_state_equations
        """

        '''
        evaluates the mass flow equations,
        modes:
            forwardpass -> calculates the loss for given variables
            jacobian -> retruns loss and jacobian of the loss with respect to the mass flows
        dem_mf_vals: 
            tensorflow variable, mass flow values ordered the same way as self.Q_heat, indexed by self.dem_ind!
        '''
        # alisases:
        A = self.A
        A_active = self.A_active
        demands = self.demands
        mf = self.mf

        A_red = tf.sparse.slice(A, start=[0, 0], size=self.A_red_size)
        A_ext = tf.concat([tf.sparse.to_dense(A_red), tf.sparse.to_dense(A_active)], axis=0)

        if not hasattr(self, 'bs'):
            bs = tf.Variable(tf.zeros((self.A_red_size[0] + A_active.get_shape()[0], 1), dtype=tf.float64))
            self.bs = bs
        else:
            bs = self.bs

        d_count = bs.get_shape()[0] - self.n_active_edges + 1
        for i in range(self.n_edges):
            edg = self.edges[i]
            if edg['edge control type'] == 'active':
                if edg['index'] in demands.keys():
                    mf_dem = dem_mf_vals[self.dem_ind[edg['index']]]
                    bs[d_count].assign([mf_dem])
                    d_count += 1

        if tf.executing_eagerly():
            warnings.warn('mass flow equations evaluated eagerly!')
        if tf.executing_eagerly():
            persistence = True
        else:
            persistence = False
        with tf.GradientTape(persistent=persistence) as tape:
            loss = (tf.matmul(A_ext, mf, a_is_sparse=True)) - bs
            for c in cycles:
                l = 0
                for e, direction in c:
                    i = self.find_edge[e]
                    edg = self.edges[i]
                    if direction == 'forward':
                        l += self.eq_pipe_p(mf[i], edg['fd_nom'], edg['length [m]'] * edg['bend_factor'], edg['diameter'])
                    else:
                        l -= self.eq_pipe_p(mf[i], edg['fd_nom'], edg['length [m]'] * edg['bend_factor'], edg['diameter'])
                loss = tf.concat([loss, tf.expand_dims(l, axis=0)], axis=0)

        if mode == 'forwardpass':
            return loss
        elif mode == 'jacobian':
            return loss, tf.squeeze(tape.jacobian(loss, mf))



    def neighbours(self, node, relationship=None, passive_only=True) -> Tuple[Set[int], Set[int], List]:
        """
        returns the neighbours of a node
        input:
            node -> node to search the neighbours of, given by its indices
            relationship -> either 'parent' or 'child' or None -> determined in terms of mass flow direction
            passive_only -> boolean, if true, consider only nodes connected by passive edges

        returns:
            tuple consisting of node-indices, edge-indices and edges-dicts corresponding to the neighbours of the node
        """
        aim_dir = {'parent': -1, 'child': 1, None: 0}[relationship]

        neighbour_nodes_ind = set()
        neighbour_edges_ind = set()
        neighbour_edges = list()
        for edg_ind in range(len(self.edges)):
            e = self.edges[edg_ind]
            if passive_only & (e['edge control type'] == 'active'):
                continue
            if e['from'] == self.nodes[node]['index']:
                edg_dir = 1
                node_ind = self.find_node[e['to']]
            elif e['to'] == self.nodes[node]['index']:
                edg_dir = -1
                node_ind = self.find_node[e['from']]
            else:
                continue  # skip remaining loop if the node is no neighbour
            mf_dir = np.sign(self.mf[edg_ind])
            if aim_dir == 0:
                neighbour_nodes_ind.add(node_ind)
                neighbour_edges_ind.add(edg_ind)
                neighbour_edges.append(e)
            # it holds true (?), that for related edges edg_dir * aim_dir * mf_dir = 1 right and -1 for wrong direction
            # if the mass flow is zero, the product is zero and the edge is not appended
            elif edg_dir * aim_dir * mf_dir == 1:
                neighbour_nodes_ind.add(node_ind)
                neighbour_edges_ind.add(edg_ind)
                neighbour_edges.append(e)
        return neighbour_nodes_ind, neighbour_edges_ind, neighbour_edges

    def solve_temperature_fixed_mf(self):
        """
        solves the temperature loss and mixing equations for fixed mass flows
        this is done by iteratively calculating the end of line temperatures for edges, whose start node has a known
        temperature and solving the mixing equation for nodes, where the temperature of all incoming flows is known.

        If this does not converge, raise a TemperatureProbagationException
        """

        T = self.T
        mf = self.mf
        T_end = self.T_end

        # first: set T_end = T_a for all edges with zero mass flow
        zero_mfs = tf.where(mf == 0)[:, 0]
        for e_ind in zero_mfs:
            T_end[e_ind].assign([self.Ta])

        s_nodes = set(np.where(self.masks[0] == 0)[0])  # nodes, where the temperature is solved
        new_nodes = s_nodes.copy()  # nodes added in the last iteration of the loop
        # first: set T_end = T_a for all edges with zero mass flow
        while len(s_nodes) != self.n_nodes:
            # solve line temp loss for all edges connected to new_nodes
            c_nodes = set()  # candidate nodes to be solved next
            for ind in new_nodes:
                # edges_fr = [e for e in self.edges if (e['from'] == self.nodes[ind]['index']) & (e['edge control type'] == 'passive')]
                new_c_nodes, _, edges_fr = self.neighbours(ind, 'child')
                c_nodes.update(new_c_nodes)
                for edg in edges_fr:
                    i = self.find_edge[edg['index']]
                    T_end[i, :].assign(
                        self.eq_pipe_T(tf.abs(mf[i, :]), T[ind, :], tf.constant(edg['temp_loss_coeff'], dtype=tf.float64),
                                       tf.constant(edg['length [m]'], dtype=tf.float64)))

            new_nodes = set()
            # solve temperature mixing equation for all nodes where it is possible:
            for ind in c_nodes:
                p_nodes, p_edges, _ = self.neighbours(ind, 'parent')
                if p_nodes.issubset(s_nodes):  # if the temperature for all parent nodes is known:
                    temp = tf.reduce_sum(
                        tf.math.abs(tf.gather(mf, list(p_edges), axis=0)) * tf.gather(T_end, list(p_edges), axis=0)) / \
                           tf.reduce_sum(tf.math.abs(tf.gather(mf, list(p_edges), axis=0)))
                    T[ind].assign(tf.expand_dims(temp, axis=-1))
                    # add the new node to s_nodes and new_nodes
                    new_nodes.add(ind)
            s_nodes.update(new_nodes)

            if len(new_nodes) == 0:
                if len(s_nodes) != self.n_nodes:
                    '''
                    some nodes were not reached. This is either because:
                      a) some nodes are not connected with the supplied grid
                      b) some mass flows are zero and therefore the edges are not considered in the algorithm
                      c) something went horrible wrong
                    for a and b, set temperature to self.Ta (ambient temp), for c raise Exception
                    '''
                    missing_nodes = set(range(self.n_nodes)).difference(s_nodes)
                    for ind in missing_nodes:
                        _, edg_ind, _ = self.neighbours(ind)
                        # if the mass flow in all connected edges is zero:
                        if tf.reduce_max(tf.math.abs(tf.gather(self.mf, list(edg_ind), axis=0))) < 1.e-8:
                            self.T[ind].assign([self.Ta])
                            for e_ind in edg_ind:
                                self.T_end[e_ind].assign([self.Ta])
                            s_nodes.add(ind)
                            new_nodes.add(ind)
                    if len(new_nodes) == 0:
                        raise TemperatureProbagationException()

    def solve_pressures(self, gridmodel):
        '''
            dp_ij = k * mf * |mf| -> this can be transformed into a Matrix multiplication:
            p_i = p_00 + K @ (mf . |mf|) where K contains all edges on a path between p_0 and p_i and mf is a vector

            pros: K has to be calculated only once and is constant even for cycles!
            grid is passed as nx.digraph object to create K-Matrix (way easier than with self.edges representation
        '''
        if not hasattr(self, 'dp_matrix'):
            K = np.zeros((self.n_nodes, self.n_edges))
            K_sparse_ind = []
            vals = []
            p0 = np.zeros((self.n_nodes, 1))
            passive_grid = gridmodel.get_passive_grid().to_undirected()
            # all pathes from all nodes to all nodes with fixed pressures
            paths = {}
            for f_dp in self.fix_dp.keys():
                paths[f_dp] = nx.algorithms.single_target_shortest_path(passive_grid, f_dp)
            for i, params in enumerate(self.nodes):
                # find the path to fixed dp node:
                path, fixpoint = [(paths[k].get(params['index'], ), k) for k in paths.keys() if paths[k].get(params['index'], )][0]
                # set p0:
                p0[i] = self.fix_dp[fixpoint]
                # set entries in K-matrix:
                for j in range(len(path)-1):
                    # set sign according to edge orientation
                    try:
                        edg_params = gridmodel.edges[(path[j], path[j+1])]
                        edg_ind = edg_params['index']
                        sign = 1
                    except KeyError:
                        edg_params = gridmodel.edges[(path[j+1], path[j])]
                        edg_ind = edg_params['index']
                        sign = -1
                    col = self.find_edge[edg_ind]    # col: edge position in K-matrix
                    # val: value of entry for K-matrix  - pressure loss for mf = 1  (dp ~ mf^2)
                    val = sign * self.eq_pipe_p(tf.cast(1., tf.float64), tf.cast(edg_params['fd_nom'], tf.float64),
                                                tf.cast(edg_params['length [m]'] * edg_params['bend_factor'], tf.float64),
                                                tf.cast(edg_params['diameter'], tf.float64))
                    K[i, col] = val
                    vals.append(val)
                    K_sparse_ind.append([i, col])
            Ks = tf.sparse.SparseTensor(indices=K_sparse_ind, values=vals, dense_shape=[self.n_nodes, self.n_edges])
            Ks = tf.sparse.reorder(Ks)
            self.p_0 = p0
            self.dp_matrix = Ks

        # solve pressure-equations in matrix-multiplication form
        self._solve_pressures()

    @tf.function
    def _solve_pressures(self):
        self.p.assign(self.p_0 + tf.sparse.sparse_dense_matmul(self.dp_matrix, self.mf * tf.math.abs(self.mf)))

    def evaluate_pressure_equations(self, grid):
        pass

    def add_to_variables(self, values):
        # adds the values to the not masked variables
        end = 0
        variables = [self.T, self.mf, self.p, self.T_end]
        for i in range(len(variables)):
            start = end
            end = start + int(np.sum(self.masks[i]))
            M = self.mask_matrix[i]
            variables[i].assign_add(tf.sparse.sparse_dense_matmul(M, values[start:end]))

    def report_state(self):
        # prints the current values for the state variables
        variables = [self.T, self.mf, self.p, self.T_end]
        for i in range(len(variables)):
            print(variables[i].name)
            if tf.shape(variables[i]).numpy()[0] == self.n_nodes:
                for j in range(self.n_nodes):
                    print(self.nodes[j]['index'] + '  ' + str(variables[i].numpy()[j]))
            else:
                for j in range(self.n_edges):
                    print(self.edges[j]['index'] + '  ' + str(variables[i].numpy()[j]))
            print("\n")


