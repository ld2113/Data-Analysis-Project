print("in, 1, out")
import keras
from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import pandas as pd
import sklearn.metrics as met



train = pd.read_pickle('arrays/domain_df.pckl').values


input = Input(shape=(train.shape[1],))
encoded = Dense(1, activation='relu')(input)
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
cb.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=2, verbose=1, mode='auto', epsilon=1e-07, cooldown=0, min_lr=0))

autoencoder.fit(train, train,
                epochs=4000,
                batch_size=64,
                shuffle=True,
                callbacks=cb)

print("INSUM",np.sum(train))
pred_out = autoencoder.predict(train)
bin_out = (pred_out > 0.5).astype(int)
print("OUTSUM",np.sum(bin_out))
print("Accuracy:", met.accuracy_score(train, bin_out))
print("Precision:", met.precision_score(train, bin_out, average='micro'))
print("AUROC:", met.roc_auc_score(train, pred_out, average='micro'))
print("AUPRC:", met.average_precision_score(train, pred_out, average='micro'))
