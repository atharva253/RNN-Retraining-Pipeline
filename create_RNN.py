import orchest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

hidden_units = orchest.get_step_param("hidden_units")
dense_units = orchest.get_step_param("dense_units")
time_steps = orchest.get_step_param("time_steps")
num_epochs = orchest.get_step_param("num_epochs")

input_shape = (time_steps,1)
activation = ['tanh','tanh']

data = orchest.get_inputs()
trainX, trainY, testX, testY = data["XY_data"]

model = Sequential()
model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
model.add(Dense(units=dense_units, activation=activation[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=num_epochs, batch_size=1, verbose=2)

orchest.output(model, name = "model")