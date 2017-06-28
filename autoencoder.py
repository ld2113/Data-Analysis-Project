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

import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization

def main(reg, drop, optim, batchsize, act, max_epochs, lr, train_split, lr_sched, lr_plat,
		in_path, save_path, struct):

	class Cust_metrics(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			return

		def on_train_end(self, logs={}):
			return

		def on_epoch_begin(self, epoch, logs={}):
			return

		def on_epoch_end(self, epoch, logs={}):

			if (epoch+1)%500 == 0 and epoch != 0:
				pred_train = self.model.predict(train, batch_size=batchsize, verbose=1)
				pred_train_bin = np.array(pred_train > 0.5).astype(int)

				insum_train = np.sum(train)

				outsum_train = np.sum(pred_train_bin)
				acc_train = met.accuracy_score(train,pred_train_bin)
				prec_train = met.precision_score(train, pred_train_bin, average='micro')
				recall_train = met.recall_score(train, pred_train_bin, average='micro')
				f1_train = met.f1_score(train, pred_train_bin, average='micro')
				auroc_train = met.roc_auc_score(train, pred_train, average='micro')
				aupr_train = met.average_precision_score(train, pred_train, average='micro')

				if train_split != 1.0:
					pred_test = self.model.predict(test, batch_size=batchsize, verbose=1)
					pred_test_bin = np.array(pred_test > 0.5).astype(int)

					insum_test = np.sum(test)

					outsum_test = np.sum(pred_test_bin)
					acc_test = met.accuracy_score(test,pred_test_bin)
					prec_test = met.precision_score(test, pred_test_bin, average='micro')
					recall_test = met.recall_score(test, pred_test_bin, average='micro')
					f1_test = met.f1_score(test, pred_test_bin, average='micro')
					auroc_test = met.roc_auc_score(test, pred_test, average='micro')
					aupr_test = met.average_precision_score(test, pred_test, average='micro')

					print('\n', 'insum_train:', insum_train, ' - outsum_train:', outsum_train, ' - acc_train:', acc_train, ' - prec_train:', prec_train, end=' - ')
					print('recall_train:', recall_train, ' - f1_train:', f1_train, ' - auroc_train:', auroc_train, ' - aupr_train:', aupr_train)

					print('\n', 'insum_test:', insum_test, ' - outsum_test:', outsum_test, ' - acc_test:', acc_test, ' - prec_test:', prec_test, end=' - ')
					print('recall_test:', recall_test, ' - f1_test:', f1_test, ' - auroc_test:', auroc_test, ' - aupr_test:', aupr_test)

				else:
					outsum_test = 'n/a'
					acc_test = 'n/a'
					prec_test = 'n/a'
					recall_test = 'n/a'
					f1_test = 'n/a'
					auroc_test = 'n/a'
					aupr_test = 'n/a'

					print('\n', 'insum_train:', insum, ' - outsum_train:', outsum_train, ' - acc_train:', acc_train, ' - prec_train:', prec_train, end=' - ')
					print('recall_train:', recall_train, ' - f1_train:', f1_train, ' - auroc_train:', auroc_train, ' - aupr_train:', aupr_train)

				with open('log_autoenc.csv', 'a') as f:
					f.write(','.join(list(map(lambda x: str.replace(x, ",", ";"),list(map(str,[id,time.strftime('%Y%m%d'),time.strftime('%H%M'), epoch+1, max_epochs,
					train_split, logs['loss'], logs['acc'], insum_train, outsum_train, acc_train, prec_train, recall_train, f1_train, auroc_train, aupr_train, insum_test, outsum_test, acc_test, prec_test, recall_test, f1_test, auroc_test, aupr_test,
					struct,reg,drop,batchsize,act,optim,lr,lr_sched,lr_plat,in_path,save_path,'\n']))))))

			return

		def on_batch_begin(self, batch, logs={}):
			return

		def on_batch_end(self, batch, logs={}):
			return


	################################ Functions #####################################

	def build_model(struct, trainshape, act):

		input = Input(shape=(trainshape,))

		encoded = Dense(struct[0], activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(input)
		encoded = BatchNormalization()(encoded)
		encoded = Activation(act)(encoded)
		encoded = Dropout(drop)(encoded)
		for i in range(len(struct)):
			if i == 0:
				pass
			else:
				encoded = Dense(struct[i], activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(encoded)
				encoded = BatchNormalization()(encoded)
				encoded = Activation(act)(encoded)
				encoded = Dropout(drop)(encoded)

		if len(struct) == 1:
			decoded = Dense(trainshape, activation='sigmoid')(encoded)
		else:
			decoded = Dense(struct[-2], activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(encoded)
			decoded = BatchNormalization()(decoded)
			decoded = Activation(act)(decoded)
			decoded = Dropout(drop)(decoded)
			for i in range(len(struct)):
				if i > 1:
					decoded = Dense(struct[-(i+1)], activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(decoded)
					decoded = BatchNormalization()(decoded)
					decoded = Activation(act)(decoded)
					decoded = Dropout(drop)(decoded)
			decoded = Dense(trainshape, activation='sigmoid')(decoded)

		autoencoder = Model(input, decoded)

		return autoencoder, input, encoded


	def set_callbacks():
		cb = []
		cb.append(Cust_metrics())

		if lr_sched[1] == 'pow2':
			def schedule(epoch):
				n = 0
				if epoch%lr_sched[0] == 0 and epoch != 0:
					n += 1
				return lr/pow(2,n)
			cb.append(keras.callbacks.LearningRateScheduler(schedule))

		elif lr_plat != []:
			cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=lr_plat[0], patience=lr_plat[1], verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))

		#cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode='auto'))
		#cb.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))

		return cb



############################### Main Program ###################################

	with open('counter_autoenc.txt', 'r') as f:
		id = int(f.readline()) + 1
	with open('counter_autoenc.txt', 'w') as f:
		f.write(str(id))


	################################################################
	print("----Loading Data----")

	data = pd.read_pickle(in_path).values

	if train_split == 1.0:
		train = data
		test = data
	else:
		test = data[int(train_split*len(data)):]
		train = data[0:int(train_split*len(data))]

	trainshape = train.shape[1]


	################################################################
	print("----Setting up the model----")

	model, input, encoded = build_model(struct, trainshape, act)


	################################################################
	print("----Compiling the model----")

	if optim == 'adam':
		opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	elif optim == 'sgd':
		opt = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=True)

	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


	################################################################
	print("----Model training----")

	model.fit(train, train, epochs=max_epochs, batch_size=batchsize, callbacks=set_callbacks())


	################################################################
	if save_path != '':
		print("----Writing encoding to file----")

		encoder = Model(input, encoded)
		enc_pred = encoder.predict(data)
		np.save(save_path,enc_pred)
