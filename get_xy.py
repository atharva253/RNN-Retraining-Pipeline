import orchest
import numpy as np

time_steps = orchest.get_step_param("time_steps")
data = orchest.get_inputs()
train_data, test_data = data["train_test_data"]


Y_ind_train = np.arange(time_steps, len(train_data), time_steps)
Y_train = train_data[Y_ind_train]
rows_x_train = len(Y_train)
X_train = train_data[range(time_steps*rows_x_train)]
X_train = np.reshape(X_train, (rows_x_train, time_steps, 1))

Y_ind_test = np.arange(time_steps, len(test_data), time_steps)
Y_test = test_data[Y_ind_test]
rows_x_test = len(Y_test)
X_test = test_data[range(time_steps*rows_x_test)]
X_test = np.reshape(X_test, (rows_x_test, time_steps, 1))

print("Get_XY Successful!")
orchest.output((X_train, Y_train, X_test, Y_test), name = "XY_data")
