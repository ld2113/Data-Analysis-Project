print("in, 20, out")
import keras
from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import pandas as pd
import sklearn.metrics as met



train = pd.read_pickle('arrays/domain_df.pckl').values

#train = data[0:int(0.8*len(data))]
#test = data[int(0.8*len(data)):]


input = Input(shape=(train.shape[1],))
encoded = Dense(20, activation='tanh')(input)
#encoded = Dense(1, activation='relu')(encoded)
#encoded = Dense(1024, activation='relu')(encoded)

#decoded = Dense(128, activation='relu')(encoded)
#decoded = Dense(4096, activation='relu')(decoded)
decoded = Dense(train.shape[1], activation='sigmoid')(encoded)


autoencoder = Model(input, decoded)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

cb = []
cb.append(keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'))
cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=2, verbose=1, mode='auto', epsilon=1e-08, cooldown=0, min_lr=0))

autoencoder.fit(train, train,
                epochs=4000,
                batch_size=64,
                shuffle=True,
                callbacks=cb)

print("TRAIN STATISTICS")
print("INSUM",np.sum(train))
pred_out = autoencoder.predict(train)
bin_out = (pred_out > 0.5).astype(int)
print("OUTSUM",np.sum(bin_out))
print("Accuracy:", met.accuracy_score(train, bin_out))
print("Precision:", met.precision_score(train, bin_out, average='micro'))
print("AUROC:", met.roc_auc_score(train, pred_out, average='micro'))
print("AUPRC:", met.average_precision_score(train, pred_out, average='micro'))
'''
print("TEST STATISTICS")
print("INSUM",np.sum(test))
pred_test = autoencoder.predict(test)
bin_test = (pred_test > 0.5).astype(int)
print("OUTSUM",np.sum(bin_test))
print("Accuracy:", met.accuracy_score(test, bin_test))
print("Precision:", met.precision_score(test, bin_test, average='micro'))
print("AUROC:", met.roc_auc_score(test, pred_test, average='micro'))
print("AUPRC:", met.average_precision_score(test, pred_test, average='micro'))
'''
