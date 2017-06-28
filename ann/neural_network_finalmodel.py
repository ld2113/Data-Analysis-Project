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

	class Cust_metrics(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			return

		def on_train_end(self, logs={}):
			return

		def on_epoch_begin(self, epoch, logs={}):
			return

		def on_epoch_end(self, epoch, logs={}):
			pred = self.model.predict([test_lab_names[:,0:2],test_lab_names[:,0:2],test_lab_names[:,0:2]], batch_size=batchsize, verbose=1)

			bin_pred = np.array(pred > 0.5).astype(int)
			labels_test = test_lab_names[0:pred.shape[0],2]

			acc = met.accuracy_score(labels_test,bin_pred)
			prec = met.precision_score(labels_test,bin_pred)
			recall = met.recall_score(labels_test,bin_pred)
			f1 = met.f1_score(labels_test,bin_pred)
			mcc = met.matthews_corrcoef(labels_test,bin_pred)
			auroc = met.roc_auc_score(labels_test, pred)
			aupr = met.average_precision_score(labels_test, pred)
			frac_pos = np.sum(bin_pred)/len(bin_pred)

			print('\n', 'auroc:', auroc, ' - aupr:', aupr, end=' - ')
			print('mcc:', mcc, ' - acc:', acc, ' - prec:', prec, ' - recall:', recall,' - f1:', f1, end=' - ')
			print("Frac. pos. pred.:", frac_pos)

			# with open('log_finalmodel.csv', 'a') as f:
			# 	f.write(','.join(list(map(lambda x: str.replace(x, ",", ";"),list(map(str,[id,time.strftime('%Y%m%d'),time.strftime('%H%M'),max_epochs,mcc,acc,prec,recall,f1,auroc,aupr,frac_pos,
			# 	input_mode,coord_struct,dom_struct,go_struct,concat_struct,reg,drop,batchsize,act,optim,lr,lr_sched,lr_plat,class_weight,train_path,test_path,coord_path,dom_path,go_path,coord_norm,aux_output_weights,epoch+1,'\n']))))))

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


	def build_model(input_mode, coord_struct, dom_struct, go_struct, concat_struct, coord_embedding, dom_embedding, go_embedding):


		if input_mode == 'cdg':
			coord_in = Input(shape=(2,), dtype='int32', name='coord_in')
			coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False)(coord_in)
			coord_net = Flatten()(coord_net)
			for n in coord_struct:
				coord_net = Dense(n, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(coord_net)
				coord_net = BatchNormalization()(coord_net)
				coord_net = Activation(act)(coord_net)
				coord_net = Dropout(drop)(coord_net)
			if aux_output_weights != []:
				coord_out = Dense(1, activation='sigmoid', name='coord_out')(coord_net)

			dom_in = Input(shape=(2,), dtype='int32', name='dom_in')
			dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False)(dom_in)
			dom_net = Flatten()(dom_net)
			for n in dom_struct:
				dom_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
				dom_net = BatchNormalization()(dom_net)
				dom_net = Activation(act)(dom_net)
				dom_net = Dropout(drop)(dom_net)
			if aux_output_weights != []:
				dom_out = Dense(1, activation='sigmoid', name='dom_out')(dom_net)

			go_in = Input(shape=(2,), dtype='int32', name='go_in')
			go_net = Embedding(output_dim=go_embedding.shape[1],input_dim=go_embedding.shape[0],weights=[go_embedding],input_length=2,trainable=False)(go_in)
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


		elif input_mode == 'dg':
			go_in = Input(shape=(2,), dtype='int32', name='go_in')
			go_net = Embedding(output_dim=go_embedding.shape[1],input_dim=go_embedding.shape[0],weights=[go_embedding],input_length=2,trainable=False)(go_in)
			go_net = Flatten()(go_net)
			for n in go_struct:
				go_net = Dense(n, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(go_net)
				go_net = BatchNormalization()(go_net)
				go_net = Activation(act)(go_net)
				go_net = Dropout(drop)(go_net)
			if aux_output_weights != []:
				go_out = Dense(1, activation='sigmoid', name='go_out')(go_net)

			dom_in = Input(shape=(2,), dtype='int32', name='dom_in')
			dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False)(dom_in)
			dom_net = Flatten()(dom_net)
			for n in dom_struct:
				dom_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
				dom_net = BatchNormalization()(dom_net)
				dom_net = Activation(act)(dom_net)
				dom_net = Dropout(drop)(dom_net)
			if aux_output_weights != []:
				dom_out = Dense(1, activation='sigmoid', name='dom_out')(dom_net)

			x = keras.layers.concatenate([go_net, dom_net])
			for n in concat_struct:
				x = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(x)
				x = BatchNormalization()(x)
				x = Activation(act)(x)
				x = Dropout(drop)(x)

			output = Dense(1, activation='sigmoid', name='output')(x)

			if aux_output_weights == []:
				model = Model(inputs=[dom_in, go_in], outputs=output)
			else:
				model = Model(inputs=[dom_in, go_in], outputs=[output, dom_out, go_out])


		elif input_mode == 'cd':
			coord_in = Input(shape=(2,), dtype='int32', name='coord_in')
			coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False)(coord_in)
			coord_net = Flatten()(coord_net)
			for n in coord_struct:
				coord_net = Dense(n, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(coord_net)
				coord_net = BatchNormalization()(coord_net)
				coord_net = Activation(act)(coord_net)
				coord_net = Dropout(drop)(coord_net)
			if aux_output_weights != []:
				coord_out = Dense(1, activation='sigmoid', name='coord_out')(coord_net)

			dom_in = Input(shape=(2,), dtype='int32', name='dom_in')
			dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False)(dom_in)
			dom_net = Flatten()(dom_net)
			for n in dom_struct:
				dom_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
				dom_net = BatchNormalization()(dom_net)
				dom_net = Activation(act)(dom_net)
				dom_net = Dropout(drop)(dom_net)
			if aux_output_weights != []:
				dom_out = Dense(1, activation='sigmoid', name='dom_out')(dom_net)

			x = keras.layers.concatenate([coord_net, dom_net])
			for n in concat_struct:
				x = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(x)
				x = BatchNormalization()(x)
				x = Activation(act)(x)
				x = Dropout(drop)(x)

			output = Dense(1, activation='sigmoid', name='output')(x)

			model = Model(inputs=[coord_in, dom_in], outputs=output)
			if aux_output_weights == []:
				model = Model(inputs=[coord_in, dom_in], outputs=output)
			else:
				model = Model(inputs=[coord_in, dom_in], outputs=[output, coord_out, dom_out])

		elif input_mode == 'c':
			coord_in = Input(shape=(2,), dtype='int32', name='coord_in')
			coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False)(coord_in)
			coord_net = Flatten()(coord_net)
			for n in coord_struct:
				coord_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(coord_net)
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
				dom_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(dom_net)
				dom_net = BatchNormalization()(dom_net)
				dom_net = Activation(act)(dom_net)
				dom_net = Dropout(drop)(dom_net)
			output = Dense(1, activation='sigmoid', name='output')(dom_net)

			model = Model(inputs=dom_in, outputs=output)

		elif input_mode == 'g':
			go_in = Input(shape=(2,), dtype='int32', name='go_in')
			go_net = Embedding(output_dim=go_embedding.shape[1],input_dim=go_embedding.shape[0],weights=[go_embedding],input_length=2,trainable=False)(go_in)
			go_net = Flatten()(go_net)
			for n in go_struct:
				go_net = Dense(n, activation=None, kernel_regularizer=regularizers.l2(reg), kernel_initializer='he_normal')(go_net)
				go_net = BatchNormalization()(go_net)
				go_net = Activation(act)(go_net)
				go_net = Dropout(drop)(go_net)
			output = Dense(1, activation='sigmoid', name='output')(go_net)

			model = Model(inputs=go_in, outputs=output)

		return model


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

	# with open('counter_finalmodel.txt', 'r') as f:
	# 	id = int(f.readline()) + 1
	# with open('counter_finalmodel.txt', 'w') as f:
	# 	f.write(str(id))


	################################################################
	print("----FINAL MODEL TESTING----")
	print("----Loading Data----")

	dom_embedding = np.load(dom_path)
	go_embedding = np.load(go_path)

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

	model = build_model(input_mode, coord_struct, dom_struct, go_struct, concat_struct, coord_embedding, dom_embedding, go_embedding)


	################################################################
	print("----Compiling the model----")

	if optim == 'adam':
		opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	elif optim == 'sgd':
		opt = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=True)

	if aux_output_weights == []:
		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	else:
		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], loss_weights=aux_output_weights)



	################################################################
	print("----Model training----")

	fit_in = [train_lab_names[:,0:2]] * len(input_mode)
	fit_out = [train_lab_names[:,2]] * max(1,len(aux_output_weights))
	cb=[keras.callbacks.ModelCheckpoint('finalmodel_save/model.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1),Cust_metrics()]
	model.fit(fit_in, fit_out, epochs=max_epochs, batch_size=batchsize, callbacks=cb, class_weight = class_weight)


	################################################################
	print("----Evaluating and saving model----")

	model.save("FINAL_MODEL.HDF5")

	pred = model.predict([test_lab_names[:,0:2],test_lab_names[:,0:2],test_lab_names[:,0:2]], batch_size=batchsize, verbose=1)

	bin_pred = np.array(pred > 0.5).astype(int)
	labels_test = test_lab_names[0:pred.shape[0],2]

	acc = met.accuracy_score(labels_test,bin_pred)
	prec = met.precision_score(labels_test,bin_pred)
	recall = met.recall_score(labels_test,bin_pred)
	f1 = met.f1_score(labels_test,bin_pred)
	mcc = met.matthews_corrcoef(labels_test,bin_pred)
	auroc = met.roc_auc_score(labels_test, pred)
	aupr = met.average_precision_score(labels_test, pred)
	frac_pos = np.sum(bin_pred)/len(bin_pred)

	print('\n', 'auroc:', auroc, ' - aupr:', aupr, end=' - ')
	print('mcc:', mcc, ' - acc:', acc, ' - prec:', prec, ' - recall:', recall,' - f1:', f1, end=' - ')
	print("Frac. pos. pred.:", frac_pos)

	# with open('log_finalmodel.csv', 'a') as f:
	# 	f.write(','.join(list(map(lambda x: str.replace(x, ",", ";"),list(map(str,[id,time.strftime('%Y%m%d'),time.strftime('%H%M'),max_epochs,mcc,acc,prec,recall,f1,auroc,aupr,frac_pos,
	# 	input_mode,coord_struct,dom_struct,go_struct,concat_struct,reg,drop,batchsize,act,optim,lr,lr_sched,lr_plat,class_weight,train_path,test_path,coord_path,dom_path,go_path,coord_norm,aux_output_weights,'\n']))))))
