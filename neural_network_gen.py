from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
import sys
import pickle

import sklearn.metrics as met
from imblearn.over_sampling import SMOTE
from multiprocessing_generator import ParallelGenerator

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout


class Cust_metrics(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		n_gen_iter = int(np.floor(test_lab_names.shape[0]/batchsize))
		with ParallelGenerator(test_gen(test_lab_names, coord_df, domains_df, batchsize), max_lookahead=100) as g:
			epoch_pred = self.model.predict_generator(g, n_gen_iter)
		epoch_bin_pred = np.array(epoch_pred > 0.5).astype(int)
		labels_test = test_lab_names[0:epoch_pred.shape[0],2]

		print('\n', 'Custom epoch metrics:  auroc:', met.roc_auc_score(labels_test, epoch_pred), ' - aupr:', met.average_precision_score(labels_test, epoch_pred), end=' - ')
		print('mcc:', met.matthews_corrcoef(labels_test,epoch_bin_pred), ' - acc:', met.accuracy_score(labels_test,epoch_bin_pred), end=' - ')
		print("Prediciton  MAX:", max(epoch_pred) , "Prediction MIN:", min(epoch_pred), end=' - ')
		print("Frac. pos. pred.:", np.sum(epoch_bin_pred)/len(epoch_bin_pred))

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


################################ Functions #####################################
'''
def normalise_coord(array_list):

	for i in range(len(array_list)):
		array_list[i] -= np.mean(array_list[i], axis=0)
		array_list[i] =  array_list[i] / np.std(array_list[i], axis=0)

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
'''


def test_gen(test_lab_names, coord_df, domains_df, b_size):
	c = 0
	while True:
		coord1 = coord_df.loc[test_lab_names[c*b_size:(c+1)*b_size,0]].values
		domains1 = domains_df.loc[test_lab_names[c*b_size:(c+1)*b_size,0]].values
		coord2 = coord_df.loc[test_lab_names[c*b_size:(c+1)*b_size,1]].values
		domains2 = domains_df.loc[test_lab_names[c*b_size:(c+1)*b_size,1]].values

		b_data = np.concatenate((coord1, coord2, domains1, domains2), axis=1)
		b_labels = lab_names[c*b_size:(c+1)*b_size,2]

		c += 1

		yield (b_data,b_labels)


def train_gen(lab_names_pos, lab_names_neg, coord_df, domains_df, b_size):
	while True:
		pos_sample = lab_names_pos[np.random.choice(lab_names_pos.shape[0], size=int(b_size/2), replace=False)]
		neg_sample = lab_names_neg[np.random.randint(lab_names_neg.shape[0], size=int(b_size/2))]

		lab_names_batch = np.concatenate((pos_sample, neg_sample), axis=0)
		np.random.shuffle(lab_names_batch)

		coord1 = coord_df.loc[lab_names_batch[:,0]].values
		domains1 = domains_df.loc[lab_names_batch[:,0]].values
		coord2 = coord_df.loc[lab_names_batch[:,1]].values
		domains2 = domains_df.loc[lab_names_batch[:,1]].values

		b_data = np.concatenate((coord1, coord2, domains1, domains2), axis=1)
		b_labels = lab_names_batch[:,2]

		yield (b_data,b_labels)


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
	adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#sdg = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)

	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model


def set_callbacks():

	cb = []

	#cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
	#cb.append(keras.callbacks.LearningRateScheduler(schedule))
	#cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='auto'))
	cb.append(Cust_metrics())
	#cb.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))

	return cb


def schedule(epoch):
	return 0.0003/pow(2,epoch)


############################### Main Program ###################################

# Setting hyperparameters
reg = 0.0
drop = 0.0
nlayers = 1
nunits = 138
train_split = 0.9
batchsize = 64


print("----Loading Data----")
lab_names = np.load('arrays/full_embed/names_labels_3x129m.npy')
coord = np.load('arrays/full_embed/4D_coord_4x16k.npy')
domains = np.load('arrays/encoded_dom_20_relu_1.npy')
names = np.load('arrays/names.npy')

indim = coord.shape[1] * 2 + domains.shape[1] * 2


print("----Processing Data----")
coord -= 0.5
coord *= 2.0

indx = int(train_split*len(lab_names))

coord_df = pd.DataFrame(coord, index=names)
domains_df = pd.DataFrame(domains, index=names)

train_lab_names_df = pd.DataFrame(lab_names[0:indx], index=None)
train_lab_names_pos = train_lab_names_df[train_lab_names_df[2]==1].values
train_lab_names_neg = train_lab_names_df[train_lab_names_df[2]==0].values

test_lab_names = lab_names[indx:]

#generator_train = train_gen(train_lab_names_pos, train_lab_names_neg, coord_df, domains_df, batchsize)


print("----Setting up the model----")
model = build_model(indim, reg, drop, nlayers, nunits, 'relu')


print("----Compiling the model----")
compiled_mod = compile_mod(model)


print("----Model training----")
n_gen_train = int(np.floor(indx/batchsize))
with ParallelGenerator(train_gen(train_lab_names_pos, train_lab_names_neg, coord_df, domains_df, batchsize), max_lookahead=100) as g:
	compiled_mod.fit_generator(g, n_gen_train, epochs=10, callbacks=set_callbacks(), max_q_size=100, workers=1)#,validation_split=0.25)
