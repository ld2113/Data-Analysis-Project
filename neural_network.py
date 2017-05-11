from __future__ import print_function
from __future__ import division

import numpy as np
import sys

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Load coordinate and name data
names = np.load('names.npy')
print("names", names.shape)
labels = np.load('labels.npy')
print("labels", labels.shape)
coord = np.load('4Dcoordinates.npy')

# Preprocess data (zero centered and between -1 and 1)
coord -= 0.5
coord *= 2

print("coord", coord.shape)

sys.exit()

# Setting hyperparameters
reg = 0.0
drop = 0.5

# Setting up model
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu',kernel_regularizer=regularizers.l2(reg)))
model.add(Dropout(drop))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg)))
model.add(Dropout(drop))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg)))
model.add(Dropout(drop))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(coord, labels, epochs=10, batch_size=128)

#score = model.evaluate(x_test, y_test, batch_size=128)
#print(score)
