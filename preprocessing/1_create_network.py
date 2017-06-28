from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd
import networkx as nx
import copy


def process_df(path):

	# Read Biogrid data into pandas dataframe
	df = pd.read_table(path, engine="c")

	# Rename first two columns of dataframe
	df = df.rename(columns={'#ID Interactor A': 'ID Interactor A'})
	df['ID Interactor A'] = df['ID Interactor A'].map(lambda x: x.lstrip('entrez gene/locuslink:'))
	df['ID Interactor B'] = df['ID Interactor B'].map(lambda x: x.lstrip('entrez gene/locuslink:'))

	# Remove non-human ppi from the dataframe
	df = df[df['Taxid Interactor A'].isin(['taxid:9606'])]
	df = df[df['Taxid Interactor B'].isin(['taxid:9606'])]

	# Remove self interactions
	df = df[df['ID Interactor A'] != df['ID Interactor B']]

	return df


def create_graph(df, rm_frac=0.1):
	# Create networkx graph from dataframe (single edge)
	G = nx.from_pandas_dataframe(df, 'ID Interactor A', 'ID Interactor B', create_using=nx.Graph())

	if rm_frac > 0.0:
		# np.random.seed(42)
		# ind_sample = np.random.choice(np.arange(len(nx.edges(G))), size=int(rm_frac*len(nx.edges(G))), replace=False)
		# edgelist = nx.edges(G)
		# rm_edges = [edgelist[i] for i in ind_sample]
		G.remove_edges_from(list(map(tuple,np.load('arrays/08_testing/network/rm_edges_09_test.npy').astype(int).tolist())))
		rm_edges = []
	else:
		rm_edges = []

	return G, rm_edges


def calc_pathlengths(G, K, Kmax):
	# Calculate shortest pathlengths dictionary in the graph up to a threshold using networkx
	p = nx.all_pairs_shortest_path_length(G,K)

	# Reformat path lengths to array
	dist_df_orig = pd.DataFrame.from_dict(p)
	dist_df = dist_df_orig.replace(0,2)
	dist_df = dist_df.fillna(0)-Kmax
	dist_df[dist_df == -Kmax] = 0
	dist_mat = dist_df.values
	names = np.asarray(dist_df.columns).astype(str)

	return names, dist_df_orig, dist_mat


def create_labels(dist_df_orig):
	adj_df = copy.copy(dist_df_orig)
	adj_df[adj_df!=1.0] = 0
	adj_mat = adj_df.values
	indices = np.triu_indices_from(adj_mat,k=1)
	labels = adj_mat[indices]

	return labels


############################## Main Code #######################################

# Set parameters
K = 4
Kmax = 5
path = "../BIOGRID-ORGANISM-Homo_sapiens-3.4.147.mitab.txt"

# Get data
print('---Processing Data---')
df = process_df(path)

# Create graph and remove some edges
print('---Processing Graph---')
G, rm_edges = create_graph(df, rm_frac=0.1)

# Calculate distance
print('---Calculating Distances---')
names, dist_df_orig, dist_mat = calc_pathlengths(G, K, Kmax)

# Create labels
print('---Creating Labels---')
labels = create_labels(dist_df_orig)

# Save data
print('---Saving Data---')
#np.save('arrays/names.npy', names)
np.save('arrays/08_testing/network/dist_mat_4_5.npy', dist_mat)
np.save('arrays/08_testing/network/labels_100.npy', labels)
# np.save('arrays/full_embed/network/rm_edges.npy', rm_edges)
dist_df_orig.to_pickle('arrays/08_testing/network/dist_df_orig_4_5.pkl')
