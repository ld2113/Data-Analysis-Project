from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import sys
import pickle

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
		print('\n', 'Custom epoch metrics:  auc:', met.roc_auc_score(labels_test, epoch_pred), end=' - ')

		print('mcc:', met.matthews_corrcoef(labels_test,epoch_bin_pred), ' - acc:', met.accuracy_score(labels_test,epoch_bin_pred), end=' - ')

		print("Prediciton  MAX:", max(epoch_pred) , "Prediction MIN:", min(epoch_pred), end=' - ')

		print("Fraction of positive predictions:", np.sum(epoch_bin_pred)/len(epoch_bin_pred))

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


def normalise_coord(array_list):

	for i in range(len(array_list)):
		array_list[i] -= 0.5
		array_list[i] *= 2

	return array_list


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


def build_model(indim, reg, drop, nlayers, nunits, act):

	# Setting up model
	model = Sequential()

	model.add(Dense(nunits, input_dim=indim, activation=act, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
	model.add(Dropout(drop))

	for i in range(nlayers):
		model.add(Dense(nunits, activation=act, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal'))
		model.add(Dropout(drop))

	model.add(Dense(1, activation='sigmoid'))

	return model


def compile_mod(model):

	# Compiling the model
	adam = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#sdg = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)

	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model

def set_callbacks():

	cb = []

	#cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
	cb.append(keras.callbacks.LearningRateScheduler(schedule))
	#cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='auto'))
	cb.append(Cust_metrics())

	return cb

def schedule(epoch):
	return 0.0003/pow(2,epoch)


############################### Main Program ###################################

# Setting hyperparameters
reg = 0.0
drop = 0.0
nlayers = 10
nunits = 128


print("----Loading Data----")
labels_train = np.load('arrays/labels_075_resampled.npy')
labels_test = np.load('arrays/labels_025.npy')
coord_train = np.load('arrays/4D_075_cint_resampled_8x194m.npy')
coord_test = np.load('arrays/4D_025_cint_8x32m.npy')


print("----Processing Data----")
coord_test, coord_train = normalise_coord([coord_test, coord_train])
#labels_train, labels_test, coord_train, coord_test = subset_data(labels, coord, keep=1, train_split=0.75)


print("----Setting up the model----")
model = build_model(coord_train.shape[1], reg, drop, nlayers, nunits, 'relu')


print("----Compiling the model----")
compiled_mod = compile_mod(model)


print("----Model training----")
compiled_mod.fit(coord_train, labels_train, epochs=5, batch_size=64, callbacks=set_callbacks())#,validation_split=0.25)


#pred = compiled_mod.predict(coord_test,batch_size=64, verbose=0)
#print('\n','Final Prediciton Score Summary (Validation Set Size:', np.abs(len(pred)),'):' , '\n', "MAX:", max(pred),'\n' , "MIN:", min(pred),'\n')

print(compiled_mod.get_weights())
#pickle.dump(weights,open( "weights.p", "wb"))
# Evaluate model with test data
#score = model.evaluate(coord_test, coord_test, batch_size=128)
#print(score)
