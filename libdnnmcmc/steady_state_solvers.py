# -*- coding: utf-8 -*-
"""
@author: Andreas Bott

solver for district heating systems under steady state conditions

This code is highly dependent on the formulation of the state equations and can only be used in combination with the
state_equations object defined in ./state_equations.py
"""


import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import scipy.sparse.linalg as slag


def _scipy_solve_linalg_wrapper(J, lhs):
    # wrapper for scipy sparse solver for linear equations, modified to resemble behaviour of tf.linalg.solve()
    J_sp = sp.csr_matrix(J.numpy())
    b = lhs.numpy()
    return tf.expand_dims(tf.constant(slag.spsolve(-J_sp, b), dtype=tf.float64), axis=1)

class my_newton_optimizer():
    """
    performs a root search using the Newton-Raphson Algorithm

    name might be misleading - this newton_optimizer ist not general and only works for solving the complete set of
    SS-equations using the StateEquations-Object.
    """
    def __init__(self):
        pass

    @tf.function
    def do_newton_step(self, function):
        jacobian, cur_value = function.evaluate_state_equations('jacobian')
        desc_dir = self.get_desc_dir(jacobian, cur_value)
        self.armijo_step(function, cur_value, desc_dir, tf.concat(jacobian, axis=1))

    def get_desc_dir(self, jacobians, values):
        """
            solve -jacobian @ desc_dir = value, returns desc_dir
            for numerical stability: if Jacobian is not invertible, add identity matrix
        """
        # concatenate jacobians  for different variables
        J = tf.concat(jacobians, axis=1)
        cond = lambda J: tf.math.less(tf.linalg.svd(J, compute_uv=False)[-1], 1.e-10)
        body = lambda J: [tf.add(J, tf.eye(tf.shape(J)[0], dtype=tf.float64))]
        var = J
        [J] = tf.while_loop(cond, body, [var], maximum_iterations=100)
        # usualy less than 5 iterations are needed, max iter should not be reachable. However, there may be problems in
        #   extreme settings, e.g. if all demands are zero. set max-iter = 100 to avoid infinite looping.
        #   Sometimes it actually seems to be invertible anyways, deflecting the laws of mathematics
        # pass solving the linear system of equations to scipy for better utilising sparsity of J
        desc_dir = tf.py_function(func=_scipy_solve_linalg_wrapper, inp=[J, values], Tout=tf.float64)
        return desc_dir

    def armijo_step(self, function, start_value, desc_dir, jacobian, sigma_0=1., gamma=1.e-4, max_iter=46):
        '''
            using backtracking together with the amijo-condition to find stepsize
            start with stepsize sigma_0;
            gamma and max_iter for numerical stability

            armijo-condition:
            if: f(x + sigma*d) > f(x) + gamma * sigma * grad^T * d
                reduce stepsize sigma (here: half)
            else:
                add stepsize * times desc_dir to variables

            algorithmic: add max stepsize and reduce for each subsequent step to save some calculations
        '''

        # apply max stepsize:
        function.add_to_variables(desc_dir)
        # execute loss function for point of interest
        val = function.evaluate_state_equations('forwardpass')
        # local step variables:
        sigma = tf.cast(sigma_0, dtype=tf.float64)
        gamma = tf.cast(gamma, dtype=tf.float64)

        # tf.while_loop is checked before it is run first
        cond = lambda val, sigma: \
            tf.math.greater(tf.norm(val), tf.norm(start_value + gamma * sigma * jacobian @ desc_dir))
        def body(val, sigma):
            sigma = sigma / 2
            function.add_to_variables(-desc_dir*sigma)
            val = function.evaluate_state_equations('forwardpass')
            return [val, sigma]

        # the body part of the loop alters the state variables saved in the SE object.
        # writing the return of tf.while_loop to 'val', 'sigma' is syntactically required even though its unused later
        val, sigma = tf.while_loop(cond, body, [val, sigma], maximum_iterations=max_iter)


class my_fixpoint_iteratror():
    """
        performs a root search using a decomposition Fixpoint-Iteration Algorithm
    """
    def __init__(self, nr_prec=1.e-2):
        """
        :param nr_prec: termination condition for NR solver used to determine mass flows. Values below 1.e-3 are not
                        advised and might lead to long runtimes. For higher precision use the my_newton_optimizer class.
        """
        self.nr_prec = nr_prec

    def solve_mf(self, function, mf_vals=None, cycles=[], loop=None):
        """ solves the mass flow equations for given supply temperatures or mass flows at demands
            function: SE-object the mf is solved for
            mf_vals: if None: determine mf_vals at demands from Q_heat and temperatures
                     else: pass tf.variable odered the same way as SE.Q_heat to pass fixed mf-values
            loop: if True, iteratively loop over the equations until mf does no longer change
                  if loop=None: set loop=True if cycles are in the grid
        """
        if loop is None:
            loop = not cycles == []
        if not loop:   # for grids without cycles, solving the mass flows is straight forward
            mf = function.solve_massflow_fixed_temp(mf_vals, cycles)
            function.mf.assign(mf)
        else:
            if mf_vals is None:
                # calculate mass flows from demand values:
                mf_vals = tf.Variable(np.zeros_like(function.Q_heat), dtype=tf.float64)
                for d in function.demands.keys():
                    edg = function.edges[function.find_edge[d]]
                    dt = function.T[function.find_node[edg['from']]] - function.T[function.find_node[edg['to']]]
                    # use save division to set mf to zero, if supply and return temperature are equal
                    # (both reach Ta if no heat is consumed)
                    mf = tf.maximum(tf.math.divide_no_nan(function.Q_heat[function.dem_ind[d]], (function.cp * dt)), 0)
                    # mass flows below zero are not accepted for active demands -> leads to problems in temp propagation
                    mf_vals[function.dem_ind[d]].assign(tf.squeeze(mf))
            # assign mass flow values to demand mass flows
            for d in function.demands.keys():
                function.mf[function.find_edge[d]].assign([mf_vals[function.dem_ind[d]]])

            """ iterative apply NR algorithm to solve the hydraulic equations """
            for _ in range(30):
                loss = self.NR_step_mf(function, mf_vals, cycles)
                # if np.max(np.abs(mf.numpy() - last_mf)) < 1.e-10:
                if tf.reduce_sum(loss**2) < self.nr_prec:
                    break
            else:
                print('max iterations NR for mf')

    def NR_step_mf(self, function, mf_vals, cycles=[]):
        # see my_newton_optimizer for implementation details on NR algorithm
        # - here: same approach applied to mass flows only
        if not hasattr(self, '_demMask'):
            dem_mask = np.eye(function.mf.get_shape()[0])
            for d in function.demands.keys():
                ind = function.find_edge[d]
                dem_mask[ind, ind] = 0
            self._demMask = tf.constant(dem_mask, dtype=tf.float64)

        values, J = function.evaluate_mf_equations('jacobian', mf_vals, cycles)

        # calculate descent direction, add ones to diagonal if J is not invertible
        cond = lambda J: tf.math.less(tf.linalg.svd(J, compute_uv=False)[-1], 1.e-10)
        body = lambda J: [tf.add(J, tf.eye(tf.shape(J)[0], dtype=tf.float64))]
        var = J
        [J] = tf.while_loop(cond, body, [var])
        # desc_dir = tf.linalg.solve(-J, values)
        desc_dir = tf.py_function(func=_scipy_solve_linalg_wrapper, inp=[J, values], Tout=tf.float64)

        # use armijo-algorithm to determine maximum step size - check my_newoton_optimizer for details
        # apply max stepsize:
        function.mf.assign_add(tf.matmul(self._demMask, desc_dir, a_is_sparse=True))
        # execute loss function for point of interest
        val = function.evaluate_mf_equations('forwardpass', mf_vals, cycles)
        # local step variables:
        sigma = tf.constant(1., dtype=tf.float64)
        gamma = tf.constant(1.e-4, dtype=tf.float64)
        max_iter = 46
        # tf.while_loop is checked before it is run first
        cond = lambda val, sigma: \
            tf.math.greater(tf.norm(val), tf.norm(values + gamma * sigma * J @ desc_dir))

        def body(val, sigma):
            sigma = sigma / 2
            function.mf.assign_add(-1 * tf.matmul(self._demMask, desc_dir, a_is_sparse=True) * sigma)
            val = function.evaluate_mf_equations('forwardpass', mf_vals, cycles)
            return [val, sigma]

        val, sigma = tf.while_loop(cond, body, [val, sigma], maximum_iterations=max_iter)
        return val

    def solve_temp(self, function):
        function.solve_temperature_fixed_mf()

    def solve_p(self, function, grid):
        function.solve_pressures(grid)


class SE_solver():
    """
    combines the solvers above in one function to solve SE-objects' state equations
    """
    def __init__(self, nr_prec=1.e-5, mf_nr_prec=1.e-2):
        self.FI = my_fixpoint_iteratror(mf_nr_prec)
        self.NR = my_newton_optimizer()
        self.nr_prec = nr_prec
        self.mf_nr_prec = mf_nr_prec
        self.fi_prec = max(1.e-2, nr_prec)
        # there is no point in solving the fi step below the nr precision
        # 1.e-2 is experimentally proven to be a good choice for our systems

    def fixpoint_step(self, SE, cycles):
        # executes one step of the fixpoint iteration algorithm
        self.FI.solve_mf(SE, cycles=cycles)
        self.FI.solve_temp(SE)
        loss = SE.evaluate_state_equations('forwardpass')
        return loss

    def newton_raphson_step(self, SE):
        # executes one step of the newton-raphson-algorithm
        self.NR.do_newton_step(SE)
        loss = SE.evaluate_state_equations('forwardpass')
        return loss

    def __call__(self, SE, cycles, max_steps=100, verbose=False):
        # FI-steps:
        loss = 0
        last_loss = tf.Variable(np.inf, dtype=tf.float64)
        for steps in range(max_steps):
            loss = tf.reduce_sum(self.fixpoint_step(SE, cycles) ** 2)
            gain = (last_loss - loss) / loss
            if verbose:
                print(f'FI-step: {steps}; loss: {loss.numpy()}; gain: {gain.numpy()}')
            if tf.math.greater(0.2, gain) or tf.math.greater(self.fi_prec, loss):
                break
            else:
                last_loss.assign(loss)

        # NR-steps:
        stale_counter = 0
        step = 0
        while tf.math.greater(loss, self.nr_prec):
            """
            it is possible for the NR algorithm to reach region with very slow convergence. 
            If nearly no gain is achieved over 5 NR steps, one FI step is done to get out of these regions. 
            """
            if stale_counter < 5:
                loss = tf.reduce_sum(self.newton_raphson_step(SE) ** 2)
                gain = (last_loss - loss) / loss
                if verbose:
                    print(f'NR-step: {step}; stale_count: {stale_counter}; loss: {loss.numpy()}; gain: {gain.numpy()};')
                if tf.math.greater(5.e-2, gain):
                    stale_counter += 1
                else:
                    stale_counter = 0
                    last_loss.assign(loss)
            else:
                loss = tf.reduce_sum(self.fixpoint_step(SE, cycles) ** 2)
                if verbose:
                    gain = (last_loss - loss) / loss
                    print(f'FI-step: {step}; loss: {loss.numpy()}; gain: {gain.numpy()}')
                last_loss.assign(loss)
                stale_counter = 0

            if step >= max_steps:
                break
            else:
                step += 1
