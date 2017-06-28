# Create list of protein labels corresponding to embedding output and dict to link names to indices and vice versa
import pickle
import numpy as np

#dist_df_orig = pickle.load(open('arrays/090_embed/network/dist_df_orig_4_5.pkl','rb'))

#names = np.array(dist_df_orig.columns, dtype=int)

names = np.load('arrays/prediction/allnames19.npy')

index_dic = dict(enumerate(names))

inverse_dic = {v: k for k, v in index_dic.items()}

pickle.dump(index_dic,open('arrays/prediction/index_dic.pkl','wb'))
pickle.dump(inverse_dic,open('arrays/prediction/inverse_dic.pkl','wb'))
