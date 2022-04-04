import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preparation(dataframe, feature_list, target_variable,
                     test_size=0.4, random_state=42, 
                     print_shapes=False, standardize=False):
    
    X = dataframe[feature_list]
    y = dataframe[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if print_shapes == True:
        print('Data Shapes:')
        print('X_train ', X_train.shape)
        print('y_train ', y_train.shape)
        print('X_test ', X_test.shape)
        print('y_test ', y_test.shape)   

    if standardize == False:
        return X_train, X_test, y_train, y_test
    elif standardize == True:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
