"""
This script calculates an upper bound for the error due to the steady-state assumption.

The theory behind the calculations are explained in Appendix C of the paper.
"""

import numpy as np
import tensorflow as tf
from libdnnmcmc.utility import load_scenario
from libdnnmcmc.steady_state_solvers import SE_solver, my_fixpoint_iteratror


def get_flow_speed(mf):
    d = 0.0697  # m (DN65-pipe)
    rho = 997  # kg/m3
    A = d**2/4 * np.pi
    return mf / (rho * A)

#%% Testcase: Paper-Grid-Loop:
SE, VM, d_prior_dist, _ = load_scenario('loop')
cycles = VM.cycles
FI = my_fixpoint_iteratror()

solve_SE = SE_solver()
def _solve_SE(demands, SE):
    for dem in SE.demands.keys():
        ind = SE.dem_ind[dem]
        d = demands[ind]
        SE.Q_heat[ind].assign(d)
    solve_SE(SE, cycles)
    return tf.reduce_sum(SE.evaluate_state_equations('forwardpass')**2)

## workflow:
#  1 draw initial demand
#  2 solve state equations (steady state)
#  3 calculate run times
#  4 modify demand (i.e. modify mass flows at demand)
#  5 solve hydraulic system with new mass flows
#  6 solve thermal system with new mass flows
#  skip  7 save demands with new temperatures/mass flows - not promising results
#  8 solve steady state with new demands
#  9 compare thermal solution and steady-state solution; temperatures at demands
# 10 store results over multiple runs to compare probabilistic

#%%
# 1 draw inital demand:
n_runs = 50
q_init = d_prior_dist.sample(n_runs)
q_modifier = [0.5, 0.7, 1.3, 1.5]
n_modifier = len(q_modifier)

z_pattern = np.zeros((n_runs, n_modifier))
Ta_ss, Tb_ss, Tc_ss, Td_ss = np.zeros_like(z_pattern), np.zeros_like(z_pattern), np.zeros_like(z_pattern), np.zeros_like(z_pattern)
Ta_qss, Tb_qss, Tc_qss, Td_qss = np.zeros_like(z_pattern), np.zeros_like(z_pattern), np.zeros_like(z_pattern), np.zeros_like(z_pattern)

for mod_ind, modifier in enumerate(q_modifier):
    q_new = q_init * modifier

    for run_ind in range(n_runs):

        # 2. solve state equations:
        psi = _solve_SE(q_init[run_ind, :], SE)

        # if the calculations did not converge for whatever reason
        if psi > 1.e-5:
            SE.set_init_state()
            psi = _solve_SE(q_init[run_ind, :], SE)
            if psi > 1.e-5:
                raise Exception(f'System did not converge with psi = {psi:4f}')

        # 3. calculate run times:
        AD = 70 / get_flow_speed(SE.mf[SE.find_edge['ADS']])
        DC = 300 / get_flow_speed(SE.mf[SE.find_edge['DCS']])
        CG = 300 / get_flow_speed(SE.mf[SE.find_edge['CGS']])
        DH = 300 / get_flow_speed(SE.mf[SE.find_edge['DHS']])
        HG = 300 / get_flow_speed(SE.mf[SE.find_edge['HGS']])
        CB = 70 / get_flow_speed(SE.mf[SE.find_edge['CBS']])
        DE = 70 / get_flow_speed(SE.mf[SE.find_edge['DES']])
        GF = 70 / get_flow_speed(SE.mf[SE.find_edge['GFS']])
        HI = 70 / get_flow_speed(SE.mf[SE.find_edge['HIS']])

        if CG > 0 and HG > 0:
            tA = AD + DC + CB
            tB = AD + tf.reduce_min([DC + CG, DH + HG]) + GF
            tC = AD + DE
            tD = AD + DH + HI
        elif CG > 0:
            tA = AD + DC + CB
            tB = AD + DC + CG + GF
            tC = AD + DE
            tD = AD + DH + HI
        else:
            tA = AD + DC + CB
            tB = AD + DH + HG + GF
            tC = AD + DE
            tD = AD + DH + HI

        print(f'run times: \n A: {tA.numpy()[0]/60:2f} min \n B: {tB.numpy()[0]/60:2f} min \n '
              f'C: {tC.numpy()[0]/60:2f} min \n D: {tD.numpy()[0]/60:2f} min')

        if tA < 0 or tB < 0 or tC < 0 or tD < 0:
            print('negative run time')

        # 4. modify mass flows according to demand:
        mf_init = tf.concat([SE.mf[SE.find_edge[k]] for k in SE.demands.keys()], axis=0)
        mf_new = mf_init * modifier

        #  5 solve hydraulic system with new mass flows
        FI.solve_mf(SE, mf_new, cycles)

        #  6 solve thermal system with new mass flows
        FI.solve_temp(SE)
        TA_qss = SE.T[SE.find_node['BS']]
        TB_qss = SE.T[SE.find_node['FS']]
        TC_qss = SE.T[SE.find_node['ES']]
        TD_qss = SE.T[SE.find_node['IS']]
        Ta_qss [run_ind, mod_ind], Tb_qss[run_ind, mod_ind], Tc_qss[run_ind, mod_ind], Td_qss[run_ind, mod_ind] = TA_qss, TB_qss, TC_qss, TD_qss

        #  7 save demands with new temperatures/mass flows???
        SE.set_demand_from_grid()

        #  8 solve steady state with new demands
        _solve_SE(q_new[run_ind,:], SE)

        #  9 compare thermal solution and steady-state solution; temperatures at demands
        TA_ss = SE.T[SE.find_node['BS']]
        TB_ss = SE.T[SE.find_node['FS']]
        TC_ss = SE.T[SE.find_node['ES']]
        TD_ss = SE.T[SE.find_node['IS']]
        Ta_ss [run_ind, mod_ind], Tb_ss[run_ind, mod_ind], Tc_ss[run_ind, mod_ind], Td_ss[run_ind, mod_ind] = TA_ss, TB_ss, TC_ss, TD_ss

        print(f'T_diff Dem A: {TA_ss - TA_qss} \n '
              f'T_diff Dem B: {TB_ss - TB_qss} \n '
              f'T_diff Dem c: {TC_ss - TC_qss} \n '
              f'T_diff Dem D: {TD_ss - TD_qss} \n ')


print(f'########################## probabilistic comaprison: ############################## \n')
for mod_ind, mod in enumerate(q_modifier):
    # 10 compare results over multiple runs:
    diff_a = Ta_ss[:, mod_ind] - Ta_qss[:, mod_ind]
    diff_b = Tb_ss[:, mod_ind] - Tb_qss[:, mod_ind]
    diff_c = Tc_ss[:, mod_ind] - Tc_qss[:, mod_ind]
    diff_d = Td_ss[:, mod_ind] - Td_qss[:, mod_ind]
    print(f'q_modifier: {mod} \n'
          f'T_diff Dem A: mean: {np.mean(np.abs(diff_a)):2f}, std: {np.std(diff_a):2f} \n'
          f'T_diff Dem B: mean: {np.mean(np.abs(diff_b)):2f}, std: {np.std(diff_b):2f} \n'
          f'T_diff Dem C: mean: {np.mean(np.abs(diff_c)):2f}, std: {np.std(diff_c):2f} \n'
          f'T_diff Dem D: mean: {np.mean(np.abs(diff_d)):2f}, std: {np.std(diff_d):2f}')
