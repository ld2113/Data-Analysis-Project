from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pickle


def indexify(list, dic):

	for arr in list:
		arr[:,0:2] = np.vectorize(dic.get)(arr[:,0:2])

	return list


###### Load Data #######
train_lab_names_pos = np.load('arrays/090_embed/nl_pos_train_01.npy')
train_lab_names_neg = np.load('arrays/090_embed/nl_neg_train_01.npy')
test_lab_names = np.load('arrays/090_embed/nl_test_01.npy')
dic = pickle.load(open('arrays/inverse_dic.pkl','rb'))

###### Process Data ######
list = [train_lab_names_pos,train_lab_names_neg,test_lab_names]
list = indexify(list, dic)

###### Save Data ######
np.save('arrays/090_embed/nl_pos_train_01_emb.npy',list[0].astype(int))
np.save('arrays/090_embed/nl_neg_train_01_emb.npy',list[1].astype(int))
np.save('arrays/090_embed/nl_test_01_emb.npy',list[2].astype(int))
