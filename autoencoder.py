import keras
from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import pandas as pd
import sklearn.metrics as met



train = pd.read_pickle('arrays/domain_df.pckl').values


input = Input(shape=(train.shape[1],))
encoded = Dense(1024, activation='relu')(input)
#encoded = Dense(2048, activation='relu')(encoded)
#encoded = Dense(1024, activation='relu')(encoded)

#decoded = Dense(2048, activation='relu')(encoded)
#decoded = Dense(4096, activation='relu')(decoded)
decoded = Dense(train.shape[1], activation='sigmoid')(encoded)


autoencoder = Model(input, decoded)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(train, train,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_split=0.25)

print("INSUM",np.sum(train))
bin_out = (autoencoder.predict(train) > 0.5).astype(int)
print("OUTSUM",np.sum(bin_out))

#print("AUROC", met.roc_auc_score(labels_test, epoch_pred))
