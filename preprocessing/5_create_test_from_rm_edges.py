from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd


def split_sets(nam_lab_full, rm_edges, factor):

	rm = list(map(tuple,list(map(sorted, rm_edges))))
	d = dict(zip(list(map(tuple,list(map(sorted,nam_lab_full[:,0:2])))), np.squeeze(nam_lab_full[:,2:3]).tolist()))

	for item in rm:
		del d[item]

	df = pd.DataFrame.from_dict(d, orient='index')

	df = pd.DataFrame(np.concatenate(( np.asarray((df.index.values.tolist())) , np.expand_dims(df[0].values, axis=1) ),axis=1))

	pos_train = df[df[2]==1].values
	pos_test = np.concatenate((rm_edges,np.expand_dims(np.ones(len(rm_edges), dtype=int), axis=1)), axis=1)

	cut = abs(int(factor*len(nam_lab_full) - len(rm_edges)))
	neg_train = df[df[2]==0].values

	np.random.shuffle(neg_train)

	neg_test = neg_train[-cut:]
	neg_train = neg_train[:-cut]

	return pos_train, neg_train, pos_test, neg_test



###### Load Data #######
nam_lab_full = np.concatenate((np.load('arrays/names_inter_2x129m.npy'), np.expand_dims(np.load('arrays/075_embed/network/labels_100.npy'), axis=1)), axis=1)
rm_edges = np.load('arrays/075_embed/network/rm_edges.npy').astype(int)

###### Process Data ######
pos_train, neg_train, pos_test, neg_test = split_sets(nam_lab_full, rm_edges, 0.1)

###### Save Data ######
np.save('arrays/075_embed/nl_pos_train_01.npy',pos_train)
np.save('arrays/075_embed/nl_neg_train_01.npy',neg_train)
np.save('arrays/075_embed/nl_pos_test_01.npy',pos_test)
np.save('arrays/075_embed/nl_neg_test_01.npy',neg_test)
