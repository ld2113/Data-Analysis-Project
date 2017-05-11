##Summary:
#Total proteins in Biogrid DB (only human-human interaction):                                                16,109
#Total number of unique interactions in Biogrid DB (only human-human interaction and no self-interactions): 219,216
#Total number of interactions in Biogrid DB (only human-human interaction and no self-interactions):        301,448
#Max possible interactions between 16,109 proteins (excluding self interactions):                       129,741,886
#Number of elements in adjacency matrix:                                                                259,499,881

from __future__ import print_function
from __future__ import division

import scipy.spatial
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

from time import time
import sys
import copy



np.random.seed(100)
K = 4
Kmax = 5
dimen = 2

t1 = time()

# Read Biogrid data into pandas dataframe
df = pd.read_table("~/work/BIOGRID-ORGANISM-Homo_sapiens-3.4.147.mitab.txt", engine="c")

# Rename first two columns of dataframe
df = df.rename(columns={'#ID Interactor A': 'ID Interactor A'})
df['ID Interactor A'] = df['ID Interactor A'].map(lambda x: x.lstrip('entrez gene/locuslink:'))
df['ID Interactor B'] = df['ID Interactor B'].map(lambda x: x.lstrip('entrez gene/locuslink:'))


# Remove non-human ppi from the dataframe
df = df[df['Taxid Interactor A'].isin(['taxid:9606'])]
df = df[df['Taxid Interactor B'].isin(['taxid:9606'])]

# Remove self interactions
df = df[df['ID Interactor A'] != df['ID Interactor B']]

# Create networkx graph from dataframe (single edge)
G = nx.from_pandas_dataframe(df, 'ID Interactor A', 'ID Interactor B', create_using=nx.Graph())

# Determine total number of proteins in the network
N = G.number_of_nodes()

# Calculate shortest pathlengths dictionary in the graph up to a threshold using networkx
p = nx.all_pairs_shortest_path_length(G,K)
t2 = time()

print("Start until shortest pathlengths:",t2-t1)

# Reformat path lengths to array
dist_df_orig = pd.DataFrame.from_dict(p)
dist_df = dist_df_orig.replace(0,2)
dist_df = dist_df.fillna(0)-Kmax
dist_df[dist_df == -Kmax] = 0
dist_mat = dist_df.values

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

t3=time()

print("Shortest pathlength up to while loop:", t3-t2)

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
	#print(vdiffmax)

	for i in range(dimen):
		v[i] = vnew[i]


for i in range(dimen):
	mv[i] = np.mean(v[i])

	Av[i] = -0.5*(dist_mat.dot(v[i]) - mv[i]*sum + (mv[i]*sumsum/N - (np.dot(sum.T,v[i])*(1/N)).item()*np.ones((N,1))))
	lam[i] = np.dot(v[i].T,Av[i])

	xvals[:,i] = np.squeeze(np.sqrt(lam[i]) * v[i])
	xvals[:,i] -= xvals[:,i].min()
	xvals[:,i] /= xvals[:,i].max()

t4=time()

print("While loop until end:", t4-t3)

print("Total time:", t4-t1)

print(xvals)


# Create list of protein labels corresponding to embedding output and dict to link names to indices and vice versa
names = np.array(dist_df_orig.columns, dtype=int)
index_dic = dict(enumerate(names))
inverse_dic = {v: k for k, v in index_dic.items()}

# Create adjacency df from distance df and convert to 1D
adj_df = copy.copy(dist_df_orig)
adj_df[adj_df!=1.0] = 0
adj_mat = adj_df.values
indices = np.triu_indices_from(adj_mat,k=1)
one_d = adj_mat[indices]

# Calculate distances between all points
distances = scipy.spatial.distance.cdist(xvals,xvals)
one_d_dist = distances[np.triu_indices_from(distances,k=1)]

# Number of zero distances
print("Number of zero distances: ",len(one_d[np.nonzero(one_d_dist==0)]))

# Calculate metrics (threshold independent)
inv_one_d_dist = 1/one_d_dist
inv_one_d_dist[inv_one_d_dist == np.inf] = 0
print("ROC AUC: ",sklearn.metrics.roc_auc_score(one_d,inv_one_d_dist))
print("PRC AUC: ",sklearn.metrics.average_precision_score(one_d,inv_one_d_dist))

# Replace distances with binary labels according to a threshold
one_d_thresh_ind = one_d_dist > 0.1
one_d_dist_bin = copy.copy(one_d_dist)
one_d_dist_bin[one_d_thresh_ind] = 0
one_d_dist_bin[np.invert(one_d_thresh_ind)] = 1

# Calculate proportion of 1 in embedded labels
#dict(zip(np.unique(one_d_dist_bin, return_counts=True)[0],np.unique(one_d_dist_bin, return_counts=True)[1]))
print("Number of predicted interactions at this threshold: ", np.sum(one_d_dist_bin))

# Calculate metrics (threshold dependent)

print("Precision: ",sklearn.metrics.precision_score(one_d,one_d_dist_bin))
print("Accuracy: ",sklearn.metrics.accuracy_score(one_d,one_d_dist_bin))
print("Matthews Corr Coeff: ", sklearn.metrics.matthews_corrcoef(one_d,one_d_dist_bin))

# Plot 2D embedding
plot = plt.scatter(xvals[:,0],xvals[:,1])
plt.show()

# Plot ROCurve
roc=sklearn.metrics.roc_curve(one_d,inv_one_d_dist)
plt.plot(roc[0],roc[1])
plt.show()

# Plot PRCurve
prc=sklearn.metrics.precision_recall_curve(one_d,inv_one_d_dist)
plt.plot(prc[0],prc[1])
plt.show()

# Plot 3D embedding
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(xvals[:,0], xvals[:,1], xvals[:,2], s=0.1)
#
# plt.show()
