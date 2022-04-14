import data_preparation
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump
import sklearn

def train_model(X_train, X_test, y_train, y_test, feature_list, print_metric=False, print_importances=False, save_model=True):

    rfm = RandomForestRegressor(random_state=0, verbose=0)
    rfm.fit(X_train, y_train)

    if print_metric == True:
        rfm_pred = rfm.predict(X_test)

        rfm_mae = mean_absolute_error(y_test, rfm_pred)
        print('MAE for Random Forest: ', round(rfm_mae, 5))

    if print_importances == True:
        rf_dict = {'Feature':np.asarray(feature_list), 'Random Forest Importance':rfm.feature_importances_}

        rfimp_df = pd.DataFrame(rf_dict, index=None)
        rfimp_df.set_index('Feature', inplace=True)
        rfimp_df.apply(lambda s: s.apply('{0:.5f}'.format))
        print(rfimp_df)

    if save_model == True:
        dump(rfm, 'model.joblib')
        print('Model object saved. sklearn version:', sklearn.__version__)
        
    return rfm