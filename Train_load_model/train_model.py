import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import keras
import csv


import time

start = time.time()

#read the input data of the model to train
input_dataset = pd.read_csv("input/input.csv")

#check data has been read in properly
print (input_dataset.head())

# The output from the input dataset
output_dataset = pd.read_csv("input/output.csv")
print (output_dataset.head())

#### KERAS

# create Model
model = Sequential()

# get number of columns in the training dataset
n_cols = input_dataset.shape[1]

# Define the number of neurons. Each layer can have different number of neurons
neurons = 50

# Add as many layers of neurons to the model. The first one has to define the number of columns from input dataset
model.add(Dense(neurons, activation="relu", input_shape=(n_cols,)))
model.add(Dense(neurons, activation="relu"))
model.add(Dense(12))

# compile model using mse as a measure of model performance
model.compile(optimizer="adam", loss="mean_squared_error")
#keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# other model that might work
#model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# set early stopping. If you want to stop training after number of epoch without improvement
#early_stopping_monitor = EarlyStopping(patience=10)

# save the trained model in a .hdf5 to solve other datasets with a trained model
checkpointer = keras.callbacks.callbacks.ModelCheckpoint(filepath="input/weights.hdf5", verbose=1, save_best_only=True)

#train the model
model.fit(input_dataset, output_dataset, validation_split=0.2, epochs=10000, callbacks=[checkpointer])
#model.fit(trainX, train_y, validation_split=0.2, epochs=300, callbacks=[early_stopping_monitor])

# Define new input data to predict
predictThis = pd.read_csv("input/input_to_predict.csv")

# After the model has been trained predict with this function
test_y_predictions = model.predict(predictThis)



# test_y_predictions is a numpy array
print(type(test_y_predictions))

# This will transform the array into a list of lists containing the results
test_list_numpy = test_y_predictions.tolist()



# From here down I just convert the tested output to a csv and the results into integers
# This is just because this case study outputs integers
test_list = []

for list_ in test_list_numpy:
    temp_list = [int(x) for x in list_]
    test_list.append(temp_list)

import csv
with open("input/prediction.csv", 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for rows in test_list:
        writer.writerow(rows)

print(test_list)
end = time.time()

print("time: {}".format(end-start))
