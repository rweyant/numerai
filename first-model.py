from keras.layers import Input, Dense, Convolution1D, Flatten, Reshape, MaxPooling1D, Dropout
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import csv
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
os.chdir('/Users/rweyant/Documents/projects/numerai/')

## LOAD DATA
print('Loading Data')
training = genfromtxt('numerai_datasets/numerai_training_data.csv', delimiter = ',', skip_header=1)
X = np.delete(training, -1, axis=1)
X = X - X.mean() 
y = training[:,-1]

# EDA
X[:,:].mean()

print('(Training) X: %s, Y: %s' % (X.shape, y.shape))

model = Sequential()
model.add(Dense(200, activation='linear', input_shape=(X.shape[1],), W_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear', W_regularizer=l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear', W_regularizer=l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear', W_regularizer=l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear', W_regularizer=l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear', W_regularizer=l2(0.05)))
#model.add(Dropout(0.2))
#model.add(Dense(21, activation='linear', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
#model.add(Dense(21, activation='linear', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

## COMPILATION
#sgd = SGD(lr=1e-2, decay=.05, momentum=0.9)
sgd = Adadelta()
model.compile(optimizer=sgd, loss='binary_crossentropy')

## Callback functions
# This will write the model after each epoch is fit, so if you kill the process
# before it finishes, you will have the output of the last completed epoch
checkpoint = ModelCheckpoint(filepath='test-checkpoint.h5')

# Fit the model
model.fit(X,
          y,
          nb_epoch=1000,
          batch_size=256,
          verbose=1,
          shuffle=True,
          validation_split=0.1,
          callbacks=[checkpoint])

####


# competition
competition = genfromtxt('numerai_datasets/numerai_tournament_data.csv', delimiter = ',', skip_header=1)
X_comp = np.delete(competition, 0, axis=1)
X_comp = X_comp - X_comp.mean() 
predictions = model.predict(X_comp)
nppred = np.asarray(predictions)
output = np.vstack((competition[:,0], nppred[:,0])).T

output_df = pd.DataFrame(output, columns = ['t_id', 'probability'])
output_df.t_id = output_df.t_id.astype(int)

output_df.to_csv('predictions.csv', index = False, header = ["\"t_id\"", "\"probability\""], quoting=csv.QUOTE_NONE)