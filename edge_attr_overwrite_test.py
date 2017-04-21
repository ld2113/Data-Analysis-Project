from __future__ import print_function
from __future__ import division

import pandas as pd
import networkx as nx
# Cannot import pyplot, _tkinter not found
#import matplotlib.pyplot as plt


df = pd.DataFrame({ 'A' : [1,2,1],
                 'B' : [5,7,5],
                 'D' : ['M1','M1','M2']})


print(df.head())

G = nx.from_pandas_dataframe(df, 'A', 'B', edge_attr=['D'], create_using=None)

print(df.groupby(['A', 'B']).count())
