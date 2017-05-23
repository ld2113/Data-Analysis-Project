import numpy as np

def create_interactions(input):
	shape = int((input.shape[0]*input.shape[0]-input.shape[0])/2.0)
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
input = np.load('../arrays/7D_coord_4x16k.npy')

# Create interactions
interactions = create_interactions(input)

# Save to file
np.save('../arrays/7D_cint_14x129m.npy', interactions)
