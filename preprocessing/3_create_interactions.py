from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


def create_interactions(input):
	# IS THE SHAPE CALCULATION CORRECT HERE??
	shape = int((input.shape[0]*input.shape[0]-input.shape[0])/2.0)
	print("SHAPE", shape)
	interactions = np.zeros([shape,input.shape[1]*2])

	# First protein coordinates
	count = 0
	for i in range(0,input.shape[0]):
		for j in range(0,input.shape[0]-(1+i)):
			interactions[count,0:input.shape[1]] = input[i]
			count += 1

	# Second protein coordinates
	count = 0
	for i in range(0,input.shape[0]):
		for j in range(0+i+1,input.shape[0]):
			interactions[count,input.shape[1]:input.shape[1]*2] = input[j]
			count += 1

	return interactions


# Load input
input = np.load('arrays/075_embed/network/labels_100.npy')

input = np.expand_dims(input, axis = 1)

# Create interactions
interactions = create_interactions(input)
print(interactions.shape)
# Save to file
#np.save('arrays/full_embed/names_labels_3x129m.npy', interactions)
