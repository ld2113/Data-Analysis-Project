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
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization

def main_emb(reg, drop, optim, batchsize, act, coord_norm, max_epochs, lr, lr_sched, lr_plat,
		dom_path, coord_path, go_path, train_path, test_path, input_mode,
		coord_struct, dom_struct, go_struct, concat_struct, aux_output_weights, class_weight):


	################################ Functions #####################################

	def normalise_coord(array_list):

		for i in range(len(array_list)):
			array_list[i] -= np.mean(array_list[i], axis=0)
			array_list[i] =  array_list[i] / np.std(array_list[i], axis=0)

		return array_list


	def build_model(input_mode, coord_struct, dom_struct, go_struct, concat_struct, coord_embedding, dom_embedding, go_embedding):

		coord_in = Input(shape=(2,), dtype='int32', name='coord_in_new')
		coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False, name='embedding_1_new')(coord_in)
		coord_net = Flatten()(coord_net)
		for n in coord_struct:
			coord_net = Dense(n, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(coord_net)
			coord_net = BatchNormalization()(coord_net)
			coord_net = Activation(act)(coord_net)
			coord_net = Dropout(drop)(coord_net)
		if aux_output_weights != []:
			coord_out = Dense(1, activation='sigmoid', name='coord_out')(coord_net)

		dom_in = Input(shape=(2,), dtype='int32', name='dom_in_new')
		dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False, name='embedding_2_new')(dom_in)
		dom_net = Flatten()(dom_net)
		for n in dom_struct:
			dom_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
			dom_net = BatchNormalization()(dom_net)
			dom_net = Activation(act)(dom_net)
			dom_net = Dropout(drop)(dom_net)
		if aux_output_weights != []:
			dom_out = Dense(1, activation='sigmoid', name='dom_out')(dom_net)

		go_in = Input(shape=(2,), dtype='int32', name='go_in_new')
		go_net = Embedding(output_dim=go_embedding.shape[1],input_dim=go_embedding.shape[0],weights=[go_embedding],input_length=2,trainable=False, name='embedding_3_new')(go_in)
		go_net = Flatten()(go_net)
		for n in go_struct:
			go_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(go_net)
			go_net = BatchNormalization()(go_net)
			go_net = Activation(act)(go_net)
			go_net = Dropout(drop)(go_net)
		if aux_output_weights != []:
			go_out = Dense(1, activation='sigmoid', name='go_out')(go_net)

		x = keras.layers.concatenate([coord_net, dom_net, go_net])
		for n in concat_struct:
			x = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(x)
			x = BatchNormalization()(x)
			x = Activation(act)(x)
			x = Dropout(drop)(x)

		output = Dense(1, activation='sigmoid', name='output')(x)

		if aux_output_weights == []:
			model = Model(inputs=[coord_in, dom_in, go_in], outputs=output)
		else:
			model = Model(inputs=[coord_in, dom_in, go_in], outputs=[output, coord_out, dom_out, go_out])


		return model


############################### Main Program ###################################


	################################################################
	print("----PREDICTION----")
	print("----Loading Data----")

	dom_embedding = np.load(dom_path)
	go_embedding = np.load(go_path)

	coord_embedding = np.load(coord_path)
	test_lab_names = np.load(test_path)

####CHANGE!!!!!!#############################################################################
	model = load_model("FINAL_MODEL.HDF5")#######################################################
	model.save_weights("FINAL_WEIGHTS.HDF5")################################################
	del model#################################################################################

	################################################################
	print("----Processing Data----")

	coord_embedding -= 0.5
	coord_embedding *= 2.0


	################################################################
	print("----Setting up the model----")

	model = build_model(input_mode, coord_struct, dom_struct, go_struct, concat_struct, coord_embedding, dom_embedding, go_embedding)
	model.load_weights("FINAL_WEIGHTS.HDF5", by_name=True)
	print("WEIGTHS LOADED")
	################################################################
	print("----Compiling the model----")
	#
	# if optim == 'adam':
	# 	opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#
	# elif optim == 'sgd':
	# 	opt = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=True)
	#
	# if aux_output_weights == []:
	# 	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	# else:
	# 	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], loss_weights=aux_output_weights)
	#
	#

	################################################################
	print("----Prediction----")

	pred = model.predict([test_lab_names,test_lab_names,test_lab_names], batch_size=batchsize, verbose=1)

	bin_pred = np.array(pred > 0.5).astype(int)
	np.save('PREDICTIONS.npy',pred)
	np.save('PREDICTIONS_BIN.npy',bin_pred)
	print('FRACTION POS PREDICTIONS: ', np.sum(bin_pred)/test_lab_names.shape[0])
