from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from imblearn.over_sampling import SMOTE


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



# Load input
coord = np.load('arrays/7D_cint_14x129m.npy')
labels = np.load('arrays/labels_100.npy')

# Subset data
labels_train, labels_test, coord_train, coord_test = subset_data(labels, coord, keep=1.0, train_split=0.75)

# Resample data
labels_train, coord_train = resample_data(labels_train, coord_train)

# Save to files
np.save('arrays/7D_075_cint_resampled_14x194m.npy', coord_train)
np.save('arrays/7D_labels_075_resampled.npy', labels_train)
np.save('arrays/7D_025_cint_14x32m.npy', coord_test)
