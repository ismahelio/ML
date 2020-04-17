import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


# To use a trained model to solve other cases

# Load the data to predict
predictThis = pd.read_csv("input/input_to_predict.csv")

# The model needs to have the same shape as the trained model: same n layers, neurons, etc.
n_cols = predictThis.shape[1]

model = Sequential()
neurons = 50
model.add(Dense(neurons, activation="relu", input_shape=(n_cols,)))
model.add(Dense(neurons, activation="relu"))
model.add(Dense(12))

# Load the weights in the model
model.load_weights("input/weights.hdf5")

# Predict the dataset
test_y_predictions = model.predict(predictThis)


test_list_numpy = test_y_predictions.tolist()
test_list = []

for list_ in test_list_numpy:
    #temp_list = [int(x) for x in list_]
    temp_list = [round(x) for x in list_]

    test_list.append(temp_list)



print(test_list)
