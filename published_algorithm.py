from __future__ import print_function
from __future__ import division

import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt


# Read Biogrid data into pandas dataframe
df = pd.read_table("~/work/BIOGRID-ORGANISM-Homo_sapiens-3.4.147.mitab.txt", engine="c")

# Rename first two columns of dataframe
df = df.rename(columns={'#ID Interactor A': 'ID Interactor A'})
df['ID Interactor A'] = df['ID Interactor A'].map(lambda x: x.lstrip('entrez gene/locuslink:'))
df['ID Interactor B'] = df['ID Interactor B'].map(lambda x: x.lstrip('entrez gene/locuslink:'))

# Only use subset of dataframe to create the graph
df = df.ix[0:10]
#print(df.head())
#print(df.groupby(['ID Interactor A', 'ID Interactor B']).count())

# Create networkx graph from dataframe (single edge)
G = nx.from_pandas_dataframe(df, 'ID Interactor A', 'ID Interactor B', create_using=nx.Graph())

# Calculate shortest pathlengths dictionary in the graph up to a threshold using networkx
p = nx.all_pairs_shortest_path_length(G,4)
