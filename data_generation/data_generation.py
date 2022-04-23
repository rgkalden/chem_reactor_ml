import numpy as np
import pandas as pd

from data_generation_functions import *


def generate_dataset(params, t_eval, base_dict_1, lim_dict_1, base_dict_2, lim_dict_2,
                     num_samples, mode_1_frac, mode_2_frac, mode_label_list, random_seed=0, filename='reactor_performance_data.csv'):

    print('Generating Dataset...')
    num_samples_1 = int(num_samples * mode_1_frac)
    num_samples_2 = int(num_samples * mode_2_frac)
    num_samples_3 = num_samples - num_samples_1 - num_samples_2

    print('Generating Mode 1...')
    data_1 = generate_data(num_samples_1, base_dict_1, lim_dict_1,
                           params, t_eval, random_seed, mode_label_list[0])
    print('Generated Mode 1 data. ' + str(num_samples_1) + ' samples')

    print('Generating Mode 2...')
    data_2 = generate_data(num_samples_2, base_dict_2, lim_dict_2,
                           params, t_eval, random_seed, mode_label_list[1])
    print('Generated Mode 2 data. ' + str(num_samples_2) + ' samples')

    print('Generating Noise data...')
    data_3 = generate_noise_data(num_samples_3, base_dict_1, base_dict_2,
                                 lim_dict_1, params, t_eval, random_seed, mode_label_list[2])
    print('Generated Noise data. ' + str(num_samples_3) + ' samples')

    columns = ['Mode', 'Fao', 'Fbo', 'P', 'To', 'Cto', 'm', 'Ta', 'T_max',
               'Fa_out', 'Fb_out', 'Fc_out', 'Fd_out', 'Cc_out', 'Xa', 'Yc']

    df_1 = pd.DataFrame(data=data_1, columns=columns)
    df_2 = pd.DataFrame(data=data_2, columns=columns)
    df_3 = pd.DataFrame(data=data_3, columns=columns)

    df = pd.concat([df_1, df_2], ignore_index=True)
    df = pd.concat([df, df_3], ignore_index=True)

    # save dataframe to csv file so that results can be reproduced later
    print('Saving Dataset...')
    df.to_csv(filename, index=False)
    print('Dataset saved as ' + filename)
