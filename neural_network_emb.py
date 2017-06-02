from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
import sys
import pickle
import copy

import sklearn.metrics as met
from multiprocessing_generator import ParallelGenerator

import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Embedding


class Cust_metrics(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		epoch_pred = self.model.predict(test_lab_names[0:2])
		epoch_bin_pred = np.array(epoch_pred > 0.5).astype(int)
		labels_test = test_lab_names[0:epoch_pred.shape[0],2]

		print('\n', 'Custom epoch metrics:  auroc:', met.roc_auc_score(labels_test, epoch_pred), ' - aupr:', met.average_precision_score(labels_test, epoch_pred), end=' - ')
		print('mcc:', met.matthews_corrcoef(labels_test,epoch_bin_pred), ' - acc:', met.accuracy_score(labels_test,epoch_bin_pred), end=' - ')
		print("Prediciton  MAX:", max(epoch_pred) , "MIN:", min(epoch_pred), end=' - ')
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

'''

def test_gen(test_lab_names, coord_df, domains_df, b_size):
	c = 0
	while True:
		coord1 = coord_df.loc[test_lab_names[c*b_size:(c+1)*b_size,0]].values
		domains1 = domains_df.loc[test_lab_names[c*b_size:(c+1)*b_size,0]].values
		coord2 = coord_df.loc[test_lab_names[c*b_size:(c+1)*b_size,1]].values
		domains2 = domains_df.loc[test_lab_names[c*b_size:(c+1)*b_size,1]].values

		#b_data = np.concatenate((coord1, coord2, domains1, domains2), axis=1)
		b_coord = np.concatenate((coord1, coord2), axis=1)
		b_domains = np.concatenate((domains1, domains2), axis=1)
		b_labels = test_lab_names[c*b_size:(c+1)*b_size,2]

		c += 1

		yield ([b_coord,b_domains],b_labels)


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

		b_coord = np.concatenate((coord1, coord2), axis=1)
		b_domains = np.concatenate((domains1, domains2), axis=1)
		b_labels = lab_names_batch[:,2]

		yield ([b_coord,b_domains],b_labels)

'''

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


def load(names):
	coord_df = pd.DataFrame(np.load('arrays/090_embed/4D_coord_4x16k.npy'), index=names)

	train_lab_names_pos = np.load('arrays/090_embed/nl_pos_train_01_emb.npy')
	train_lab_names_neg = np.load('arrays/090_embed/nl_neg_train_01_emb.npy')
	test_lab_names = np.load('arrays/090_embed/nl_test_01_emb.npy')

	return coord_df, train_lab_names_pos, train_lab_names_neg, test_lab_names



############################### Main Program ###################################

# Setting hyperparameters
reg = 0.0
drop = 0.0
nlayers = 1
nunits = 138
batchsize = 64
act = 'relu'


################################################################
print("----Loading Data----")

names = np.load('arrays/names.npy')
domains_df = pd.DataFrame(np.load('arrays/encoded_dom_20_relu_1.npy'), index=names)

#Make sure that right dataset is loading (full or partial embed)!
coord_df, train_lab_names_pos, train_lab_names_neg, test_lab_names = load(names)

coord_embedding = coord_df.values
dom_embedding = domains_df.values

################################################################
print("----Processing Data----")

coord_df -= 0.5
coord_df *= 2.0

iterations = int(np.ceil((len(train_lab_names_neg)-len(train_lab_names_pos)) / len(train_lab_names_pos)))
for i in range(iterations):
	if i==0:
		train_lab_emb_pos = copy.copy(train_lab_names_pos)
	train_lab_emb_pos = np.concatenate((train_lab_emb_pos,train_lab_names_pos), axis=0)

train_lab_emb = np.concatenate((train_lab_names_neg,train_lab_emb_pos[0:len(train_lab_names_neg)]), axis=0)

np.random.shuffle(train_lab_emb)


################################################################
print("----Setting up the model----")

coord_in = Input(shape=(2,), dtype='int32', name='coord_in')
coord_net = Embedding(output_dim=coord_embedding.shape[1],input_dim=coord_embedding.shape[0],weights=[coord_embedding],input_length=2,trainable=False)(coord_in)
coord_net = Dense(128, activation=act)(coord_net)

dom_in = Input(shape=(2,), name='dom_in')
dom_net = Embedding(output_dim=dom_embedding.shape[1],input_dim=dom_embedding.shape[0],weights=[dom_embedding],input_length=2,trainable=False)(dom_in)
dom_net = Dense(128, activation=act)(dom_net)

x = keras.layers.concatenate([coord_net, dom_net])
x = Dense(128, activation=act)(x)

output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[coord_in, dom_in], outputs=output)


################################################################
print("----Compiling the model----")

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


################################################################
print("----Model training----")

model.fit([train_lab_emb[0:2], train_lab_emb[0:2]], train_lab_emb[2:3], epochs=5, batch_size=64, callbacks=set_callbacks())
