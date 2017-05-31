from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


def embedding(dimen, N, dist_mat):
	sum = np.expand_dims(np.sum(dist_mat,1),1)
	sumsum = np.sum(dist_mat)

	vdiff = np.ones((dimen,1))
	vdiffmax = 1
	count = 0

	v = []
	vnew = []
	vdiff = []
	Av = []
	mv = np.zeros(dimen)
	lam = np.zeros(dimen)
	xvals = np.zeros((N,dimen))

	for i in range(dimen):
		v.append(np.random.randn(N,1))
		vnew.append(np.zeros((N,1)))
		vdiff.append(np.zeros((N,1)))
		Av.append(np.zeros((N,1)))

	while vdiffmax > 0.001:

		for i in range(dimen):
			mv[i] = np.mean(v[i])
			vnew[i] = -0.5*(np.dot(dist_mat,v[i]) - np.multiply(mv[i],sum) + (mv[i]*sumsum/N - np.multiply(np.multiply(np.dot(sum.T,v[i]),1/N).item(),np.ones((N,1)))))

		vnew[0] /= np.linalg.norm(vnew[0],2)

		for i in range(1,dimen):
			pom = np.zeros(vnew[i].shape)

			for j in range(i):
				pom += np.dot(np.dot(vnew[j],vnew[i].T),vnew[j])

			vnew[i] -= pom
			vnew[i] /= np.linalg.norm(vnew[i],2)

		count += 1

		for i in range(dimen):
			vdiff[i] = np.linalg.norm(v[i] - vnew[i],2)

		vdiffmax = max(vdiff)
		print(vdiffmax)

		for i in range(dimen):
			v[i] = vnew[i]


	for i in range(dimen):
		mv[i] = np.mean(v[i])

		Av[i] = -0.5*(dist_mat.dot(v[i]) - mv[i]*sum + (mv[i]*sumsum/N - (np.dot(sum.T,v[i])*(1/N)).item()*np.ones((N,1))))
		lam[i] = np.dot(v[i].T,Av[i])

		xvals[:,i] = np.squeeze(np.sqrt(lam[i]) * v[i])
		xvals[:,i] -= xvals[:,i].min()
		xvals[:,i] /= xvals[:,i].max()

	return xvals


# Load data
dist_mat = np.load('arrays/075_embed/network/dist_mat_4_5.npy')

# Run embedding
dimen = 2
N = dist_mat.shape[0]
print("---Running", dimen, "dimensional Embedding---")
coord = embedding(dimen, N, dist_mat)

# Save data
np.save('arrays/075_embed/2D_coord_10x16k.npy', coord)
