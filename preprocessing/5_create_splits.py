from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd


def split_partial(nam_lab_full, rm_edges, factor):

	rm = list(map(tuple,list(map(sorted, rm_edges))))
	d = dict(zip(list(map(tuple,list(map(sorted,nam_lab_full[:,0:2])))), np.squeeze(nam_lab_full[:,2:3]).tolist()))

	for item in rm:
		del d[item]

	df = pd.DataFrame.from_dict(d, orient='index')

	df = pd.DataFrame(np.concatenate(( np.asarray((df.index.values.tolist())) , np.expand_dims(df[0].values, axis=1) ),axis=1))

	pos_train = df[df[2]==1].values
	pos_test = np.concatenate((rm_edges,np.expand_dims(np.ones(len(rm_edges), dtype=int), axis=1)), axis=1)

	neg_train = df[df[2]==0].values
	np.random.shuffle(neg_train)

	cut = abs(int(factor*len(nam_lab_full) - len(rm_edges)))
	neg_test = neg_train[-cut:]
	neg_train = neg_train[:-cut]

	train = np.concatenate((pos_train, neg_train), axis=0)
	np.random.shuffle(train)

	test = np.concatenate((pos_test[0:int(len(pos_test)/2)], neg_test[0:int(len(neg_test)/2)]), axis=0)
	val = np.concatenate((pos_test[int(len(pos_test)/2):], neg_test[int(len(neg_test)/2):]), axis=0)
	np.random.shuffle(test)
	np.random.shuffle(val)

	return pos_train, neg_train, pos_test, neg_test, train, test, val


def split_full(lab_names, train_split):

	indx = int(train_split*len(lab_names))

	train_lab_names_df = pd.DataFrame(lab_names[0:indx], index=None)
	train_lab_names_pos = train_lab_names_df[train_lab_names_df[2]==1].values
	train_lab_names_neg = train_lab_names_df[train_lab_names_df[2]==0].values

	test_lab_names = lab_names[indx:]

	return train_lab_names_pos, train_lab_names_neg, test_lab_names



###### Load Data #######
nam_lab_full = np.concatenate((np.load('arrays/names_inter_2x129m.npy'), np.expand_dims(np.load('arrays/080_embed/network/labels_100.npy'), axis=1)), axis=1)
rm_edges = np.load('arrays/080_embed/network/rm_edges.npy').astype(int)

###### Process Data ######
#pos_train, neg_train, test = split_full(nam_lab_full, train_split=0.9)
pos_train, neg_train, pos_test, neg_test, train, test, val = split_partial(nam_lab_full, rm_edges, 0.2)

###### Save Data ######
np.save('arrays/080_embed/nl_pos_train_02.npy',pos_train)
np.save('arrays/080_embed/nl_neg_train_02.npy',neg_train)
np.save('arrays/080_embed/nl_pos_test_02.npy',pos_test)
np.save('arrays/080_embed/nl_neg_test_02.npy',neg_test)
np.save('arrays/080_embed/nl_train_02.npy',train)
np.save('arrays/080_embed/nl_test_01.npy',test)
np.save('arrays/080_embed/nl_val_01.npy',val)
