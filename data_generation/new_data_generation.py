import numpy as np
import pandas as pd

import data_generation_functions
import data_generation

# Reaction chemistry parameters
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

# Process condition parameters
# Mode 1 parameters
base_dict_1 = {'Fa': 5, 'Fb': 10, 'P': 4.92, 'To': 300, 'm': 10, 'Ta': 325}
lim_dict_1 = {'Fa': 0.1, 'Fb': 0.1, 'P': 0.1, 'To': 0.03, 'm': 0.2, 'Ta': 0.03}

# Mode 2 parameters
base_dict_2 = {'Fa': 3, 'Fb': 4, 'P': 10, 'To': 500, 'm': 5, 'Ta': 325}
lim_dict_2 = {'Fa': 0.05, 'Fb': 0.05, 'P': 0.05,
              'To': 0.015, 'm': 0.1, 'Ta': 0.015}

# Adjust process condition parameters to synthesize "new" data
factor_base_1 = 0.9
factor_lim_1 = 0.9
factor_base_2 = 1.1
factor_lim_2 = 1.1
constant_features = ['Ta', 'm']

base_dict_1.update((key, value*factor_base_1) for key,
                   value in base_dict_1.items() if key not in constant_features)
base_dict_2.update((key, value*factor_base_2) for key,
                   value in base_dict_2.items() if key not in constant_features)
lim_dict_1.update((key, value*factor_lim_1)
                  for key, value in lim_dict_1.items() if key not in constant_features)
lim_dict_2.update((key, value*factor_lim_2)
                  for key, value in lim_dict_2.items() if key not in constant_features)

# Set parameters for new dataset size (note changes from original training data)
num_samples = 5000
mode_1_frac = 0.5
mode_2_frac = 0.5

# Mode 1, Mode 2, Mode 3 = Noise
mode_label_list = [1, 2, 3]

# run generate_dataset function
data_generation.generate_dataset(params, t_eval, base_dict_1, lim_dict_1, base_dict_2, lim_dict_2,
                                 num_samples, mode_1_frac, mode_2_frac, mode_label_list, filename='new_data.csv')
