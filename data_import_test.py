from __future__ import print_function
from __future__ import division

import pandas as pd
import networkx as nx
# Cannot import pyplot, _tkinter not found
#import matplotlib.pyplot as plt


df = pd.read_table("~/work/BIOGRID-ORGANISM-Homo_sapiens-3.4.147.mitab.txt", engine="c")

df = df.rename(columns={'#ID Interactor A': 'ID Interactor A'})
df['ID Interactor A'] = df['ID Interactor A'].map(lambda x: x.lstrip('entrez gene/locuslink:'))
df['ID Interactor B'] = df['ID Interactor B'].map(lambda x: x.lstrip('entrez gene/locuslink:'))

print(df.head())

G = nx.from_pandas_dataframe(df, 'ID Interactor A', 'ID Interactor B', edge_attr=['Interaction Detection Method','Interaction Types','Source Database','Interaction Identifiers','Confidence Values'], create_using=nx.Graph())

print(df.groupby(['ID Interactor A', 'ID Interactor B']).count())
