from __future__ import print_function
from __future__ import division

import numpy as np
import sys

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout

print("----Loading data----")
# Load coordinate and name data
#names = np.load('arrays/names.npy')
#print("names", names)
labels = np.load('arrays/labels.npy')
coord = np.load('arrays/coord_all_interactions.npy')

# Preprocess data (zero centered and between -1 and 1)
coord -= 0.5
coord *= 2

# Slicing the data
keep = 100000
labels = labels[0:keep]
coord = coord[0:keep]

print("----Setting up the model----")
# Setting hyperparameters
reg = 0.1
drop = 0.5

# Setting up model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu', kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
model.add(Dropout(drop))
model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
model.add(Dropout(drop))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("----Model training----")
# Model training
model.fit(coord, labels, epochs=10, batch_size=64, validation_split=0.25)

#score = model.evaluate(x_test, y_test, batch_size=128)
#print(score)
