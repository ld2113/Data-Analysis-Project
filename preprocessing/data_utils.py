# Create list of protein labels corresponding to embedding output and dict to link names to indices and vice versa
names = np.array(dist_df_orig.columns, dtype=int)
index_dic = dict(enumerate(names))
inverse_dic = {v: k for k, v in index_dic.items()}
