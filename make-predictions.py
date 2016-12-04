from keras.layers import Input, Dense, Convolution1D, Flatten, Reshape, MaxPooling1D, Dropout
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1, l2
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import csv
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd

# competition
model = load_model('test-checkpoint.h5')
competition = genfromtxt('numerai_datasets/numerai_tournament_data.csv', delimiter = ',', skip_header=1)
X_comp = np.delete(competition, 0, axis=1)
X_comp = X_comp - X_comp.mean() 
predictions = model.predict(X_comp)
nppred = np.asarray(predictions)
output = np.vstack((competition[:,0], nppred[:,0])).T

output_df = pd.DataFrame(output, columns = ['t_id', 'probability'])
output_df.t_id = output_df.t_id.astype(int)

output_df.to_csv('predictions.csv', index = False, header = ["\"t_id\"", "\"probability\""], quoting=csv.QUOTE_NONE)