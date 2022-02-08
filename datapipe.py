import orchest
inputs = orchest.get_inputs()
trainX, trainY, testX, testY = inputs["XY_data"]
orchest.output((trainX, trainY, testX, testY), name = "XY_data")