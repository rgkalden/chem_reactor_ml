import numpy as np
import pandas as pd
from joblib import load
import os

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

dirpath = os.getcwd()
print("dirpath = ", dirpath)

output_path = os.path.join(dirpath, 'output.csv')
print(output_path)

model_path = MODEL_PATH
model = load(model_path)
print('Model loaded from', model_path)

data_path = 'new_data.csv'
df = pd.read_csv(data_path)
print('New data loaded from', data_path)

feature_list = ['Fao', 'Fbo', 'P', 'To', 'Cto', 'm', 'Ta']
X = df[feature_list]
print(len(X), 'observations found')

y = model.predict(X)
print('Predictions generated for new data')

predictions_file = 'predictions.csv'
predictions = pd.DataFrame(y).to_csv(output_path, index=False, header=None)
print('Predictions saved as', output_path)
