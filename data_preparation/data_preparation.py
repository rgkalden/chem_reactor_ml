import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# input parameter variables
feature_list = ['Fao', 'Fbo', 'P', 'To', 'Cto', 'm', 'Ta']
target_variable = 'Yc'
dataframe = pd.read_csv('data_generation/reactor_performance_data.csv')

def data_preparation(dataframe, feature_list, target_variable, test_size=0.4, random_state=42, print_shapes=False):
    
    X = dataframe[feature_list]
    y = dataframe[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if print_shapes == True:
        print('Data Shapes:')
        print('X_train ', X_train.shape)
        print('y_train ', y_train.shape)
        print('X_test ', X_test.shape)
        print('y_test ', y_test.shape)

    return X_train, X_test, y_train, y_test

# Test code
X_train, X_test, y_train, y_test = data_preparation(dataframe, 
                                                    feature_list, 
                                                    target_variable, 
                                                    test_size=0.4, 
                                                    random_state=42,
                                                    print_shapes=True)

print(X_train.head())
print(y_train.head())