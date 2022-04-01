"""
Import the functions:

odes(V, x)
gen_rand_num(mean, std)
generate_data(num_samples, base_dict, lim_dict)

also import:
params dictionary
t_eval array

* function odes has been updated from Part 1 to 
  include exception handling to avoid overflow errors
  in expressions for k2c, k1a
"""

import numpy as np
from scipy.integrate import solve_ivp
import math

params = {'V': 20,
         'E2': 14000,
         'k2c_const': 40,
         'y': 1,
         'R': 1.987,
         'To': 300,
         'E1': 8000,
         'Cto': 0.2,
         'Cpco': 18.02,
         'm': 10,
         'k1a_const': 40,
         'DH1b': -10000,
         'DH2a': -8000,
         'Cpd': 16,
         'Cpa': 10,
         'Cpb': 12,
         'Cpc': 14,
         'Ua': 80}

t_eval = np.linspace(0, params['V'], params['V'] + 1)

def odes(V, x):
    try:
        # assign variables that vary with respect to V as values from list x
        Fa = x[0]
        Fb = x[1]
        Fc = x[2]
        Fd = x[3]
        T = x[4]
        Ta = x[5]        
        # explicit equations and constants
        E2 = params['E2']
        y = params['y']
        R = params['R']
        Ft = Fa + Fb + Fc + Fd
        To = params['To']
        k2c = params['k2c_const'] * math.exp((E2/R)*(1/300 - 1/T))
        E1 = params['E1']
        Cto = params['Cto']
        Ca = Cto * (Fa/Ft) * (To/T) * y    
        Cc = Cto * (Fc/Ft) * (To/T) * y
        r2c = -k2c * Ca**2 * Cc**3
        Cpco = params['Cpco']
        m = params['m']
        Cb = Cto * (Fb/Ft) * (To/T) * y
        k1a = params['k1a_const'] * math.exp((E1/R)*(1/300 - 1/T))
        r1a = -k1a * Ca * Cb**2
        r1b = 2 * r1a
        rb = r1b
        r2a = 2/3 * r2c
        DH1b = params['DH1b']
        DH2a = params['DH2a']
        r1c = -r1a
        Cpd = params['Cpd']
        Cpa = params['Cpa']
        Cpb = params['Cpb']
        Cpc = params['Cpc']
        sumFiCpi = Cpa * Fa + Cpb * Fb + Cpc * Fc + Cpd * Fd
        rc = r1c + r2c
        Ua = params['Ua']
        r2d = -3/3 * r2c # -1/3 changed to -3/3
        ra = r1a + r2a
        rd = r2d
        Qg = r1b * DH1b + r2a * DH2a
        Qr = Ua * (T - Ta)    
        # define each ODE
        dFa_dV = ra
        dFb_dV = rb
        dFc_dV = rc
        dFd_dV = rd
        dT_dV = (Qg - Qr) / sumFiCpi
        dTa_dV = Ua * (T - Ta) / (m * Cpco)

        return [dFa_dV, dFb_dV, dFc_dV, dFd_dV, dT_dV, dTa_dV]
    except:
        pass

def gen_rand_num(mean, std):
    return float(np.random.normal(mean, std, 1))


def generate_data(num_samples, base_dict, lim_dict):
    
    sample_list = []
    
    for i in range(0, num_samples):
        
        # Flowrates
        Fao = gen_rand_num(base_dict['Fa'], base_dict['Fa'] * lim_dict['Fa'])
        Fbo = gen_rand_num(base_dict['Fb'], base_dict['Fb'] * lim_dict['Fb'])

        # Pressure
        P = gen_rand_num(base_dict['P'], base_dict['P'] * lim_dict['P'])

        # Inlet Temperature
        To = gen_rand_num(base_dict['To'], base_dict['To'] * lim_dict['To'])

        # Inlet Total Concentration
        Cto = P / (0.082 * To)

        # Cooling medium flowrate
        m = gen_rand_num(base_dict['m'], base_dict['m'] * lim_dict['m'])

        # Inlet cooling medium temperature
        Ta = gen_rand_num(base_dict['Ta'], base_dict['Ta'] * lim_dict['Ta'])

        # update params dictionary
        params['To'] = To
        params['Cto'] = Cto
        params['m'] = m

        # update ODE initial conditions
        #     initial conditions, co-current heat exchange
        #    [Fa, Fb, Fc, Fd, T, Ta]
        xo = [Fao, Fbo, 0, 0, To, Ta]

        # Solve ODEs
        solution = solve_ivp(odes, [0, params['V']], xo, t_eval = t_eval)

        # Calculate outlet values
        Fao = solution.y[0][0]
        Fa_out = solution.y[0][-1]
        Fb_out = solution.y[1][-1]
        Fc_out = solution.y[2][-1]
        Fd_out = solution.y[3][-1]
        F_out = Fa_out + Fb_out + Fc_out + Fd_out
        To = xo[4]
        T_out = solution.y[4][-1]
        
        T_max = max(solution.y[4])

        Cc_out = params['Cto'] * (Fc_out / F_out) * (To / T_out)
        Xa = (Fao - Fa_out) / Fao
        #S_cd = Fc_out / Fd_out
        Yc = Fc_out / (Fao - Fa_out)
        
        sample_list.append([Fao, Fbo, P, To, Cto, m, Ta, T_max, Fa_out, Fb_out, Fc_out, Fd_out, Cc_out, Xa, Yc])
        
    data = np.asarray(sample_list)
    
    return data