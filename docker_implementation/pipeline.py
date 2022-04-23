import os.path
import numpy as np
import pandas as pd

# Check to add module folders to python path, so the python interpreter can find modules to import
#import sys
# sys.path.append(r'C:\Users\rgkal\Documents\chem_reactor_ml\data_generation')
# sys.path.append(r'C:\Users\rgkal\Documents\chem_reactor_ml\data_preparation')
# sys.path.append(r'C:\Users\rgkal\Documents\chem_reactor_ml\model_training')
# print(sys.path)

import data_generation
import data_preparation
import model_training

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

# Set parameters for dataset size
num_samples = 10000
mode_1_frac = 0.6
mode_2_frac = 0.3

# Mode 1, Mode 2, Mode 3 = Noise
mode_label_list = [1, 2, 3]

# Model Training parameters
feature_list = ['Fao', 'Fbo', 'P', 'To', 'Cto', 'm', 'Ta']
target_variable = 'Yc'
test_size = 0.4

# dataset filepath, if data has been generated already
data_filepath_1 = 'reactor_performance_data.csv'

if os.path.exists(data_filepath_1) == True:
    print('Dataset found, loading from', data_filepath_1)
    dataframe = pd.read_csv(data_filepath_1)
else:
    print('Dataset not found, generating...')
    data_generation.generate_dataset(params, t_eval, base_dict_1, lim_dict_1, base_dict_2, lim_dict_2,
                                     num_samples, mode_1_frac, mode_2_frac, mode_label_list, filename='reactor_performance_data.csv')
    dataframe = pd.read_csv('reactor_performance_data.csv')
    print('Dataset generated')

print('Preparing Data...')
X_train, X_test, y_train, y_test = data_preparation.data_preparation(dataframe, feature_list, target_variable,
                                                                     test_size=0.4, random_state=42,
                                                                     print_shapes=False, standardize=False)

print('Model Training...')
model = model_training.train_model(X_train, X_test, y_train, y_test, feature_list,
                                   print_metric=True, print_importances=True, save_model=True)
