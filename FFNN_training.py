from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.constraints import max_norm, MinMaxNorm
#from sklearn.utils import shuffle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import os

## Settings #############################################

events = 30000

px = 16
py = 16
clr = 1

## Load training data ###################################
X_train = np.ndarray(shape=(events,px,py,clr))
Y_train = np.ndarray(shape=(events,3))

fq = open('../data/All_events_Q_training.dat','r')
fy = open('../data/All_data_coordinates_training.dat','r')
fqa = open('../data/All_events_augmented_Q_training.dat','r') # augmented
fya = open('../data/All_data_coordinates_training.dat','r') # augmented

## Augmented coordinates are basically the same as non-augmented. I just had some errors popping up so I
## decided to go with separate variables and loading

## Load input matrices ######################################

print('Reading matrices...')

l=0
ln=0
j=0
for line in fq.readlines():
     line = line.strip()
     col = line.split() #list

     i=0

     for cl in col:
         if (float(cl)>0.001):
             X_train[l][i][j] = float(cl)
         else:
             X_train[l][i][j] = 0
         i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0

print("Charge matrices loaded...")

## Load outputs #############################################

i=0
l=0
for line in fy.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_train[l][i] = (float(cl)/840.0 + 1)*0.5 
         i=i+1

     l=l+1

print("Coordinates loaded...")

## Load augmented matrices ##################################

l=15000 ## starts where the loading of ordinary data stoppped
ln=0
j=0
for line in fqa.readlines():
     line = line.strip()
     col = line.split() #list

     i=0

     for cl in col:
        X_train[l][i][j] = float(cl) ## no need to condition. Done while augmenting.
        i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0

print("Augmented charge matrices loaded...")

## Load outputs ############################################

i=0
l=15000 ## starts where the loading of ordinary data stoppped
for line in fya.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_train[l][i] = (float(cl)/840.0 + 1)*0.5 
         i=i+1

     l=l+1

print("Augmented coordinates loaded...")

## Shuffle data ############################################

#X_train, Y_train = shuffle(X_train, Y_train)

# This is done so that the validation set doesn't bias towards
# augmented data while training.

## Model ###################################################

model = Sequential()

model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(448, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(464, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(96, activation='relu', kernel_constraint=max_norm(2)))
model.add(Dropout(0.1))
model.add(Dense(144, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.00032)
model.compile(loss='mse',optimizer=opt, metrics=['mse'])

## Training ################################################

history = model.fit(X_train, Y_train, batch_size = 256,
                    validation_split = 0.2, epochs = 50)

predictions = model.predict(X_train)

############################################################

# Save model to JSON
model_json = model.to_json()
with open("model_FFNN_my.json", "w") as json_file:
     json_file.write(model_json)
#Save weights to HDF5
model.save_weights("model_FFNN_my.h5")
print("Saved model to disk")

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.savefig('loss_ffnn.pdf')


