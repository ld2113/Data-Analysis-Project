from __future__ import print_function
from __future__ import division

import scipy.sparse as sp
import numpy as np

import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

K = 30
Kmax = 200
dimen = 2

# Read Biogrid data into pandas dataframe
df = pd.read_table("~/work/example_data.txt", engine="c")

# Determine total number of proteins in the network
N = df.max().max()
#N = len(np.unique(np.concatenate([df['ID Interactor A'].unique(),df['ID Interactor B'].unique()])))

print(df.head())

# Create networkx graph from dataframe (single edge)
G = nx.from_pandas_dataframe(df, 'ID Interactor A', 'ID Interactor B', create_using=nx.Graph())

# Calculate shortest pathlengths dictionary in the graph up to a threshold using networkx
p = nx.all_pairs_shortest_path_length(G,K)

# Reformat path lengths to sparse matrix (issue with self-interactions!)
dist_df = pd.DataFrame.from_dict(p).fillna(0)-Kmax
dist_df[dist_df == -Kmax] = 0
dist_mat = sp.csr_matrix(dist_df.values)



sum = dist_mat.sum(1)
sumsum = dist_mat.sum()

vdiff = np.ones((dimen,1))
vdiffmx = 1
count = 0

v = []
vnew = []
mv = np.zeros(dimen)

for i in range(dimen):
	v.append(np.random.randn(N))
	vnew.append(np.zeros(N))

while vdiffmax > 0.001:

	for i in range(dimen):
		mv[i] = np.mean(v[i])
		vnew[i] = -0.5*(dist_mat.dot(v[i]) - (mv[i]*sum).T + (mv[i]*sumsum/N - (np.dot(sum.T,v[i])*(1/N)).item()*np.ones((1,N)))).T

	vnew[0] /= np.linalg.norm(vnew[0],2)
