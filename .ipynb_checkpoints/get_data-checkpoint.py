import orchest
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

## Get the parameters
split_percent = orchest.get_step_param("split_percent")
filename = orchest.get_step_param("filename")

df = read_csv(filename, usecols=[1], engine='python')
data = np.array(df.values.astype('float32'))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data).flatten()
n = len(data)

## Point for splitting data into train and test
split = int(n*split_percent)
train_data = data[range(split)]
test_data = data[split:]

print("Get Data Successful!")
orchest.output((train_data, test_data), name = "train_test_data")

