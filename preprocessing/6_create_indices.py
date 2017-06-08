from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pickle
import copy


def indexify(list, dic):

	for arr in list:
		arr[:,0:2] = np.vectorize(dic.get)(arr[:,0:2])

	return list


def resample(train_lab_names_neg, train_lab_names_pos):

	iterations = int(np.ceil((len(train_lab_names_neg)-len(train_lab_names_pos)) / len(train_lab_names_pos)))
	for i in range(iterations):
		if i==0:
			train_lab_emb_pos = copy.copy(train_lab_names_pos)
		train_lab_emb_pos = np.concatenate((train_lab_emb_pos,train_lab_names_pos), axis=0)

	train_lab_emb = np.concatenate((train_lab_names_neg,train_lab_emb_pos[0:len(train_lab_names_neg)]), axis=0)

	np.random.shuffle(train_lab_emb)

	return train_lab_emb


###### Load Data #######
#train_lab_names_pos = np.load('arrays/090_embed/nl_pos_train_01_emb.npy')
#train_lab_names_neg = np.load('arrays/090_embed/nl_neg_train_01_emb.npy')
dic = pickle.load(open('arrays/inverse_dic.pkl','rb'))

train = np.load('arrays/080_embed/nl_train_02.npy')
test = np.load('arrays/080_embed/nl_test_01.npy')
val = np.load('arrays/080_embed/nl_val_01.npy')


###### Process Data ######
list = [train, test, val]
list = indexify(list, dic)

#train_lab_emb = resample(train_lab_names_neg, train_lab_names_pos)


###### Save Data ######
np.save('arrays/080_embed/nl_train_02_emb.npy',list[0].astype(int))
np.save('arrays/080_embed/nl_test_01_emb.npy',list[1].astype(int))
np.save('arrays/080_embed/nl_val_01_emb.npy',list[2].astype(int))
#np.save('arrays/090_embed/nl_train_res_01_emb.npy',train_lab_emb)
