from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
import sys
import pickle
import copy
import time

import sklearn.metrics as met
from multiprocessing_generator import ParallelGenerator

import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization


class Cust_metrics(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		epoch_pred = self.model.predict([test_lab_names[:,0:2],test_lab_names[:,0:2]], batch_size=batchsize, verbose=1)
		epoch_bin_pred = np.array(epoch_pred > 0.5).astype(int)
		labels_test = test_lab_names[0:epoch_pred.shape[0],2]

		acc = met.accuracy_score(labels_test,epoch_bin_pred)
		prec = met.precision_score(labels_test,epoch_bin_pred)
		mcc = met.matthews_corrcoef(labels_test,epoch_bin_pred)
		auroc = met.roc_auc_score(labels_test, epoch_pred)
		aupr = met.average_precision_score(labels_test, epoch_pred)
		frac_pos = np.sum(epoch_bin_pred)/len(epoch_bin_pred)

		print('\n', 'auroc:', auroc, ' - aupr:', aupr, end=' - ')
		print('mcc:', mcc, ' - acc:', acc, ' - prec:', prec, end=' - ')
		print("Frac. pos. pred.:", frac_pos)

		with open('log.csv', 'a') as f:
			f.write(','.join(list(map(lambda x: str.replace(x, ",", ";"),list(map(str,[id,time.strftime('%Y%m%d'),time.strftime('%H%M'),epoch+1,max_epochs,logs['loss'],logs['acc'],mcc,acc,prec,auroc,aupr,frac_pos,
			input_mode,coord_struct,dom_struct,concat_struct,reg,drop,batchsize,act,optim,lr,lr_sched,lr_plat,train_path,test_path,coord_path,dom_path,coord_norm,aux_output_weights,'\n']))))))

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


def build_model(input_mode, coord_struct, dom_struct, concat_struct, coord_embedding, dom_embedding):

	if input_mode == 'cd':
		coord_in = Input(shape=(2,), dtype='int32', name='coord_in')
		coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False)(coord_in)
		coord_net = Flatten()(coord_net)
		for n in coord_struct:
			coord_net = Dense(n, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(coord_net)
			coord_net = BatchNormalization()(coord_net)
			coord_net = Activation(act)(coord_net)
			coord_net = Dropout(drop)(coord_net)

		dom_in = Input(shape=(2,), dtype='int32', name='dom_in')
		dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False)(dom_in)
		dom_net = Flatten()(dom_net)
		for n in dom_struct:
			dom_net = Dense(n, activation=act, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
			dom_net = BatchNormalization()(dom_net)
			dom_net = Activation(act)(dom_net)
			dom_net = Dropout(drop)(dom_net)

		x = keras.layers.concatenate([coord_net, dom_net])
		for n in concat_struct:
			x = Dense(n, activation=act, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(x)
			x = BatchNormalization()(x)
			x = Activation(act)(x)
			x = Dropout(drop)(x)

		output = Dense(1, activation='sigmoid', name='output')(x)

		model = Model(inputs=[coord_in, dom_in], outputs=output)

	elif input_mode == 'c':
		coord_in = Input(shape=(2,), dtype='int32', name='coord_in')
		coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False)(coord_in)
		coord_net = Flatten()(coord_net)
		for n in coord_struct:
			coord_net = Dense(n, activation=act, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(coord_net)
			coord_net = BatchNormalization()(coord_net)
			coord_net = Activation(act)(coord_net)
			coord_net = Dropout(drop)(coord_net)
		output = Dense(1, activation='sigmoid', name='output')(coord_net)

		model = Model(inputs=coord_in, outputs=output)

	elif input_mode == 'd':
		dom_in = Input(shape=(2,), dtype='int32', name='dom_in')
		dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False)(dom_in)
		dom_net = Flatten()(dom_net)
		for n in dom_struct:
			dom_net = Dense(n, activation=act, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
			dom_net = BatchNormalization()(dom_net)
			dom_net = Activation(act)(dom_net)
			dom_net = Dropout(drop)(dom_net)
		output = Dense(1, activation='sigmoid', name='output')(dom_net)

		model = Model(inputs=dom_in, outputs=output)

	return model


def set_callbacks():
	cb = []
	cb.append(Cust_metrics())

	if lr_sched == 'pow2':
		def schedule(epoch):
			return lr/pow(2,epoch)
		cb.append(keras.callbacks.LearningRateScheduler(schedule))

	elif lr_plat != []:
		cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=lr_plat[0], patience=lr_plat[1], verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))

	#cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='auto'))
	#cb.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))

	return cb




############################### Main Program ###################################

# Setting hyperparameters
reg = 0.0
drop = 0.0
optim = 'adam' # 'adam'-> default lr:0.001; 'sgd'-> default lr:0.01
batchsize = 64
act = 'relu'
coord_norm = 'man' # 'man'-> -0.5,*2; 'mean'-> -mean, /std; None
max_epochs = 5

lr = 0.001
lr_sched = 'pow2' # 'pow2'-> lr/pow(2,epoch)
lr_plat = [] # ReduceLROnPlateau: [factor, patience]

dom_path = 'arrays/encoded_dom_20_relu_1.npy'
coord_path = 'arrays/090_embed/4D_coord_4x16k.npy'
train_path = 'arrays/090_embed/nl_train_res_01_emb.npy'
test_path = 'arrays/090_embed/nl_test_01_emb.npy'

input_mode = 'cd' # 'c'-> coordinates only; 'd'-> domains only; 'cd'-> both
coord_struct = [128,1]
dom_struct = [128,1]
concat_struct = []
aux_output_weights = []

with open('counter.txt', 'r') as f:
	id = int(f.readline()) + 1
with open('counter.txt', 'w') as f:
	f.write(str(id))


################################################################
print("----Loading Data----")

dom_embedding = np.load(dom_path)

coord_embedding = np.load(coord_path)
test_lab_names = np.load(test_path)
train_lab_names = np.load(train_path)


################################################################
print("----Processing Data----")

if coord_norm == 'man':
	coord_embedding -= 0.5
	coord_embedding *= 2.0

elif coord_norm == 'mean':
	coord_embedding = normalise_coord([coord_embedding])[0]

################################################################
print("----Setting up the model----")

model = build_model(input_mode, coord_struct, dom_struct, concat_struct, coord_embedding, dom_embedding)


################################################################
print("----Compiling the model----")

if optim == 'adam':
	opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

elif optim == 'sgd':
	opt = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


################################################################
print("----Model training----")

model.fit([train_lab_names[:,0:2], train_lab_names[:,0:2]], train_lab_names[:,2], epochs=max_epochs, batch_size=batchsize, callbacks=set_callbacks())
