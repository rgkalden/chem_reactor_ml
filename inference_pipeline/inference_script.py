# Run script from project root folder

import numpy as np
import pandas as pd
from joblib import load

model_path = 'model_training_pipeline/model.joblib'
model = load(model_path)
print('Model loaded from', model_path)

data_path = 'data_generation/new_data.csv'
df = pd.read_csv(data_path)
print('New data loaded from', data_path)

feature_list = ['Fao', 'Fbo', 'P', 'To', 'Cto', 'm', 'Ta']
X = df[feature_list]
print(len(X), 'observations found')

y = model.predict(X)
print('Predictions generated for new data')

predictions_file = 'inference_pipeline/predictions.csv'
predictions = pd.DataFrame(y).to_csv(predictions_file, index=False, header=None)
print('Predictions saved as', predictions_file)