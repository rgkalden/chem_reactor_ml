# Run script from project root folder

import numpy as np
import pandas as pd
from joblib import load

model_path = 'model.joblib'
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
predictions = pd.DataFrame(y).to_csv(predictions_file, index=False)
print('Predictions saved as', predictions_file)