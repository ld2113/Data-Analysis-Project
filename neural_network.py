from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import sys

import sklearn.metrics as met

import keras

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout


class Cust_metrics(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.aucs = []
		#self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		epoch_pred = self.model.predict(coord_test)
		epoch_bin_pred = np.array(epoch_pred > 0.5).astype(int)
		#self.losses.append(logs.get('loss'))

		#self.aucs.append(roc_auc_score(labels_test, y_pred))
		print('\n', 'Custom epoch metrics: ', 'auc: ', met.roc_auc_score(labels_test, epoch_pred), end=' - ')

		print('mcc: ', met.matthews_corrcoef(labels_test,epoch_bin_pred))

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


def prepare_data(labels_path, coord_path):

	# Load coordinate and labels data
	labels = np.load(labels_path)
	coord = np.load(coord_path)
	#names = np.load('arrays/names.npy')

	# Preprocess coordinate data (zero centered and between -1 and 1)
	coord -= 0.5
	coord *= 2

	return labels, coord


def subset_data(labels, coord, keep=1.0, train_split=1.0):

	#Subsetting data
	labels = labels[0:int(keep*len(labels))]
	coord = coord[0:int(keep*len(coord))]

	# Splitting data
	labels_train = labels[0:int(len(labels)*train_split)]
	coord_train = coord[0:int(len(coord)*train_split)]
	labels_test = labels[int(len(labels)*train_split):]
	coord_test = coord[int(len(coord)*train_split):]

	return labels_train, labels_test, coord_train, coord_test


def build_model(indim, reg, drop, nlayers, nunits):

	# Setting up model
	model = Sequential()

	model.add(Dense(nunits, input_dim=indim, activation='relu', kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
	model.add(Dropout(drop))

	for i in range(nlayers):
		model.add(Dense(nunits, activation='relu', kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
		model.add(Dropout(drop))

	model.add(Dense(1, activation='sigmoid'))

	return model


def compile_mod(model):

	# Compiling the model
	adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#sdg = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)

	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model

def set_callbacks():

	cb = []

	cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
	#cb.append(keras.callbacks.LearningRateScheduler(schedule))
	cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='auto'))
	cb.append(Cust_metrics())

	return cb


############################### Main Program ###################################

print("----Loading and processing data----")
labels, coord = prepare_data('arrays/labels.npy', 'arrays/coord_all_interactions.npy')
labels_train, labels_test, coord_train, coord_test = subset_data(labels, coord, keep=0.001, train_split=0.75)


print("----Setting up the model----")
# Setting hyperparameters
reg = 0.1
drop = 0.5
nlayers = 2
nunits = 8

model = build_model(coord.shape[1], reg, drop, nlayers, nunits)


print("----Compiling the model----")
compiled_mod = compile_mod(model)


print("----Model training----")
compiled_mod.fit(coord_train, labels_train, epochs=10, batch_size=64, callbacks=set_callbacks())#,validation_split=0.25)


pred = compiled_mod.predict(coord_test,batch_size=64, verbose=0)
print('\n','Final Prediciton Score Summary: ', '\n', "MAX:", max(pred),'\n' , "MIN:", min(pred),'\n')

# Evaluate model with test data
#score = model.evaluate(coord_test, coord_test, batch_size=128)
#print(score)
