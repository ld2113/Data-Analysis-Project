from __future__ import print_function
from __future__ import division

import numpy as np
import sys

import sklearn.metrics

import keras

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

#Subsetting data
keep=10000000
labels = labels[0:keep]
coord = coord[0:keep]

# Splitting data
train = 0.75
labels_train = labels[0:int(len(labels)*train)]
coord_train = coord[0:int(len(coord)*train)]
labels_test = labels[int(len(labels)*train):]
coord_test = coord[int(len(coord)*train):]



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
model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
model.add(Dropout(drop))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sdg = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

cb = []
cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='auto'))
#cb.append(keras.callbacks.LearningRateScheduler(schedule))


print("----Model training----")
# Model training
model.fit(coord_train, labels_train, epochs=60, batch_size=64, callbacks=cb)#,validation_split=0.25)

#score = model.evaluate(coord_test, coord_test, batch_size=128)
print("\n",max(model.predict(coord_test,batch_size=64, verbose=1)))
#print(score)
