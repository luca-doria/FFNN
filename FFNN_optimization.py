from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
from keras.constraints import MinMaxNorm, max_norm
from keras.utils import vis_utils, plot_model, model_to_dot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import os

import pickle

## Settings #############################################

events = 15000

px = 16
py = 16
clr = 1

## Load training data ###################################
X_train = np.ndarray(shape=(events,px,py,clr))
Y_train = np.ndarray(shape=(events,3))

fq = open('Data/All_events_Q_training.dat','r')
fy = open('Data/All_data_coordinates_training.dat','r')
#fqa = open('Data/All_events_augmented_Q_training.dat','r') # augmented
#fya = open('Data/All_data_coordinates_training.dat','r') # augmented

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

'''
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
'''
## Shuffle data ############################################

X_train, Y_train = shuffle(X_train, Y_train)

# This is done so that the validation set doesn't bias towards
# augmented data while training.

## Building a model #####################################

LOG_DIR=f"{int(time.time())}"

##

def build_model(hp):
    model=Sequential()
    
    model.add(Flatten())
    model.add(Dropout(hp.Float('drop_in', 0, 0.2, step=0.1)))

    for j in range(hp.Int("d_layers", 1, 4)):     ## choosing the number of dense layers and neurons in each
        model.add(Dense(units=hp.Int('dens_'+str(j),min_value=16,max_value=512,step=16), 
                        activation='relu',kernel_constraint=max_norm(hp.Int('kc_'+str(j),1,4,1))))
        model.add(Dropout(hp.Float('drop_'+str(j), 0, 0.7, step=0.1)))

    model.add(Dense(3, activation='sigmoid')) 

    opt = keras.optimizers.Adam(hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling='LOG')) ##choosing a learning rate
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])

    return model

## choices can be different - int, float, choice, etc. Tutorials and documentation cover them all. I needed just int and float for my model

## Building the tuner ##
tuner=RandomSearch(
    build_model, 
    objective="val_mse", ## what is the aim to minimize (or maximize if it's accuracy)
    max_trials=2, # number of different combinations
    executions_per_trial=1, # one can put 2, 3, ... to be sure it wasn't a lucky guess
    directory=os.path.normpath('C:/Users/Goran/Desktop/HiWi/Work/FFNN/Optimization/'),
    project_name=LOG_DIR ## for me the project name was time.time() from beginning of this cell
    )

## Parameters of the search ##

tuner.search(x=X_train,
             y=Y_train,
             epochs=10,
             batch_size=256,
             validation_split=0.2,
             verbose=1,
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', mode='min', patience=15)])

#tuner.search_space_summary()

## Writing the necesseray parameters into a file which can be read later

with open(f"C:/Users/Goran/Desktop/HiWi/Work/FFNN/Optimization/tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)
    
## Reading the pkl file and printing the best model and scores of top 10 models with their parameters

tuner=pickle.load(open("C:/Users/Goran/Desktop/HiWi/Work/FFNN/Optimization/tuner_1611434575.pkl", "rb"))
print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())

## Comment: results of tuner search can be read only in browser-based python APIs such as Jupyter or Google Colab
