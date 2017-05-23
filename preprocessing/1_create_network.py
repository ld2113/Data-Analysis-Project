from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd
import networkx as nx


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

def calc_pathlengths(df, K, Kmax):
	# Create networkx graph from dataframe (single edge)
	G = nx.from_pandas_dataframe(df, 'ID Interactor A', 'ID Interactor B', create_using=nx.Graph())
	# Determine total number of proteins in the network
	N = G.number_of_nodes()
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


# Set parameters
K = 15
Kmax = 16
path = "~/work/BIOGRID-ORGANISM-Homo_sapiens-3.4.147.mitab.txt"

# Get data
df = process_df(path)

# Calculate distance
names, dist_df_orig, dist_mat = calc_pathlengths(df, K, Kmax)

# Save data
np.save('arrays/names.npy', names)
np.save('arrays/dist_mat.npy', dist_mat)
dist_df_orig.to_pickle('arrays/dist_df_orig.pkl')
