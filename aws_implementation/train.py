import argparse
import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-estimators", type=int, default=100)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    args = parser.parse_args()

    train_data = pd.read_csv(os.path.join(args.train, args.train_file), engine='python')
    X_train = train_data.drop('Yc', axis=1)
    y_train = train_data['Yc']
    
    test_data = pd.read_csv(os.path.join(args.test, args.test_file), engine='python')
    X_test = test_data.drop('Yc', axis=1)
    y_test = test_data['Yc']
    
    #n_estimators = args.n_estimators

    rfm = RandomForestRegressor(random_state=0, n_estimators=args.n_estimators)
    rfm.fit(X_train, y_train)
    
    rfm_pred = rfm.predict(X_test)
    rfm_mae = mean_absolute_error(y_test, rfm_pred)
    print('MAE:', rfm_mae)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(rfm, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))