from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
import sys
import pickle

import sklearn.metrics as met
from imblearn.over_sampling import SMOTE

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

		generator_predict = gen_predict(lab_names[int(train_split*len(lab_names)):], coord_df, domains_df, batchsize)
		epoch_pred = self.model.predict_generator(generator_predict, int(np.ceil(len(lab_names[int(0.75*len(lab_names)):])/batchsize)))
		epoch_bin_pred = np.array(epoch_pred > 0.5).astype(int)
		#self.losses.append(logs.get('loss'))

		#self.aucs.append(roc_auc_score(labels_test, y_pred))
		print('\n', 'Custom epoch metrics:  auroc:', met.roc_auc_score(labels_test, epoch_pred), ' - aupr:', met.average_precision_score(labels_test, epoch_pred), end=' - ')

		print('mcc:', met.matthews_corrcoef(labels_test,epoch_bin_pred), ' - acc:', met.accuracy_score(labels_test,epoch_bin_pred), end=' - ')

		print("Prediciton  MAX:", max(epoch_pred) , "Prediction MIN:", min(epoch_pred), end=' - ')

		print("Fraction of positive predictions:", np.sum(epoch_bin_pred)/len(epoch_bin_pred))

		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


################################ Functions #####################################

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


def resample_data(labels, coord):

	sm = SMOTE(random_state=42, n_jobs=10)
	coord_resampled, labels_resampled = sm.fit_sample(coord, labels)

	return labels_resampled, coord_resampled


def gen_train(lab_names, coord_df, domains_df, mb_size, b_size):

	nmb = int(np.ceil(lab_names.shape[0]/mb_size))
	sm = SMOTE(random_state=42, n_jobs=10)

	for i in range(nmb):
		coord1 = coord_df.loc[lab_names[i*mb_size:(i+1)*mb_size,0]].values
		domains1 = domains_df.loc[lab_names[i*mb_size:(i+1)*mb_size,0]].values
		coord2 = coord_df.loc[lab_names[i*mb_size:(i+1)*mb_size,1]].values
		domains2 = domains_df.loc[lab_names[i*mb_size:(i+1)*mb_size,1]].values

		mb_data = np.concatenate((coord1, coord2, domains1, domains2), axis=1)
		mb_labels = lab_names[i*mb_size:(i+1)*mb_size,2]

		coord_resampled, labels_resampled = sm.fit_sample(mb_data, mb_labels)

		for j in range(int(np.ceil(labels_resampled.shape[0]/b_size))):
			yield (coord_resampled[j*b_size:(j+1)*b_size],labels_resampled[j*b_size:(j+1)*b_size])


def gen_predict(lab_names, coord_df, domains_df, b_size):
	for i in range(int(np.ceil(lab_names.shape[0]/b_size))):
		coord1 = coord_df.loc[lab_names[i*b_size:(i+1)*b_size,0]].values
		domains1 = domains_df.loc[lab_names[i*b_size:(i+1)*b_size,0]].values
		coord2 = coord_df.loc[lab_names[i*b_size:(i+1)*b_size,1]].values
		domains2 = domains_df.loc[lab_names[i*b_size:(i+1)*b_size,1]].values

		b_data = np.concatenate((coord1, coord2, domains1, domains2), axis=1)
		b_labels = lab_names[i*b_size:(i+1)*b_size,2]

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
indim = 7*2 + 11664*2
nlayers = 1
nunits = 1000
train_split = 0.75
batchsize = 64


print("----Loading Data----")
lab_names = np.load('arrays/full_embed/names_labels_3x129m.npy')
coord = np.load('arrays/full_embed/7D_coord_7x16k.npy')
names = np.load('arrays/embedding_and_names/names.npy')
domains_df = pd.read_pickle('arrays/domain_df.pckl')


print("----Processing Data----")
coord -= 0.5
coord *= 2.0
coord_df = pd.DataFrame(coord, index=names)

generator_train = gen_train(lab_names[0:int(train_split*len(lab_names))], coord_df, domains_df, 100000, 64)


print("----Setting up the model----")
model = build_model(indim, reg, drop, nlayers, nunits, 'relu')


print("----Compiling the model----")
compiled_mod = compile_mod(model)


print("----Model training----")
compiled_mod.fit_generator(generator_train, int(np.ceil(len(lab_names[0:int(train_split*len(lab_names))])/batchsize)), epochs=5, callbacks=set_callbacks(), max_q_size=100, workers=1)#,validation_split=0.25)


int(np.ceil(len(lab_names[0:int(train_split*len(lab_names))])/batchsize))


#pred = compiled_mod.predict(coord_test,batch_size=64, verbose=0)
#print('\n','Final Prediciton Score Summary (Validation Set Size:', np.abs(len(pred)),'):' , '\n', "MAX:", max(pred),'\n' , "MIN:", min(pred),'\n')

print(compiled_mod.get_weights())
#pickle.dump(weights,open( "weights.p", "wb"))
# Evaluate model with test data
#score = model.evaluate(coord_test, coord_test, batch_size=128)
#print(score)
