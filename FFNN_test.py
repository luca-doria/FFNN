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

from scipy.optimize import curve_fit

## Settings #############################################

events_test = 4999 # another 4999 in augmented

px = 16 #matrix X pixels
py = 16 #matrix Y pixels
clr = 1

## Load training data ###################################

X_test = np.ndarray(shape=(events_test,px,py, clr))
Y_test = np.ndarray(shape=(events_test,3))

fq_test = open('../data/All_events_Q_test.dat','r')
fy_test = open('../data/All_data_coordinates_test.dat','r')
fqa_test = open('../data/All_events_augmented_Q_test.dat','r')
fya_test = open('../data/All_data_coordinates_test.dat','r')

## Load regular input matrices ##############################

l=0
ln=0
j=0
for line in fq_test.readlines():
     line = line.strip()
     col = line.split()

     i=0

     for cl in col:
         if (float(cl)>0.001):
             X_test[l][i][j] = float(cl)
         else:
             X_test[l][i][j] = 0
         i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0
         
print("Charge test matrices loaded..")

## Load regular output matrices ##################################

i=0
l=0
for line in fy_test.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_test[l][i] = (float(cl.replace(',','.'))/840.0 + 1)*0.5
         i=i+1

     l=l+1

print("Test coordinates loaded..")

'''
## Load augmented input matrices ##################################

l=4999
ln=0
j=0
for line in fqa_test.readlines():
     line = line.strip()
     col = line.split()

     i=0

     for cl in col:
         if (float(cl)>0.001):
             X_test[l][i][j] = float(cl)
         else:
             X_test[l][i][j] = 0
         i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0
         
print("Charge augmented test matrices loaded..")

## Load augmented coordinates #######################################

i=0
l=4999
for line in fya_test.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_test[l][i] = (float(cl.replace(',','.'))/840.0 + 1)*0.5
         i=i+1

     l=l+1

print("Test augmented coordinates loaded..")
'''

## Load model ######################################################

# load json model description and create model
json_file = open('model_FFNN_my.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_FFNN_my.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
opt = keras.optimizers.Adam(lr=0.00032)
model.compile(loss="mse", optimizer=opt)
score = model.evaluate(X_test, Y_test) #verbose=0
print((model.metrics_names[0], score))
predictions = model.predict(X_test)

## Plotting the graphs as usual #####################################

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.scatter(predictions[:,0],Y_test[:,0],s=2)

x_hist.hist(predictions[:,0], 60, histtype='stepfilled',
orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(Y_test[:,0], 60, histtype='stepfilled',
orientation='horizontal')
y_hist.invert_xaxis()

plt.savefig('x.pdf')

#plt.close()

##########

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.scatter(predictions[:,1],Y_test[:,1],s=2)

x_hist.hist(predictions[:,1], 60, histtype='stepfilled',
orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(Y_test[:,1], 60, histtype='stepfilled',
orientation='horizontal')
y_hist.invert_xaxis()

plt.savefig('y.pdf')

#plt.close()

##########

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.scatter(predictions[:,2],Y_test[:,2],s=2)

x_hist.hist(predictions[:,2], 60, histtype='stepfilled',
orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(Y_test[:,2], 60, histtype='stepfilled',
orientation='horizontal')
y_hist.invert_xaxis()

plt.savefig('z.pdf')

#plt.close()

###########

plt.close()
plt.cla()
plt.clf()


predictions[:,0] =  840*(2*predictions[:,0] - 1) ## this without neck data
Y_test[:,0]     =  840*(2*Y_test[:,0] - 1)
xd = predictions[:,0] - Y_test[:,0]

predictions[:,1] =  840*(2*predictions[:,1] - 1)
Y_test[:,1]     =  840*(2*Y_test[:,1] - 1)
yd = predictions[:,1] - Y_test[:,1]

predictions[:,2] =  840*(2*predictions[:,2] - 1)
Y_test[:,2]     =  840*(2*Y_test[:,2] - 1)
zd = predictions[:,2] - Y_test[:,2]


plt.subplot(3,1,1)
n_x, bins_x, patches = plt.hist( xd,
200,histtype='stepfilled',range=[-200,200], density=True)
(mu, sigma) = norm.fit(xd)
y = norm.pdf(bins_x, mu, sigma)
l = plt.plot(bins_x, y, 'r--', linewidth=2)
print("Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma))

plt.subplot(3,1,2)
n_y, bins_y, patches = plt.hist( yd,
200,histtype='stepfilled',range=[-200,200], density=True)
mu, sigma = norm.fit(yd)
y = norm.pdf(bins_y, mu, sigma)
l = plt.plot(bins_y, y, 'r--', linewidth=2)
print("Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma))

plt.subplot(3,1,3)
n_z, bins_z, patches = plt.hist( zd,
200,histtype='stepfilled',range=[-200,200], density=True)
mu, sigma = norm.fit(zd)
y = norm.pdf(bins_z, mu, sigma)
l = plt.plot(bins_z, y, 'r--', linewidth=2)
print("Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma))

plt.savefig('diff.pdf')

plt.close()
plt.cla()
plt.clf()

bins_x = np.delete(bins_x, 0)
bins_y = np.delete(bins_y, 0)
bins_z = np.delete(bins_z, 0)

# Found the fitting procedure on the internet and just adapted it to our needs

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

xdata = bins_x
ydata = n_x

H, A, x0, sigma = gauss_fit(xdata, ydata)
FWHM = 2.35482 * sigma

print('The offset of the gaussian baseline is', H)
print('The center of the gaussian fit is', x0)
print('The sigma of the gaussian fit is', sigma)
print('The maximum intensity of the gaussian fit is', H + A)
print('The Amplitude of the gaussian fit is', A)
print('The FWHM of the gaussian fit is', FWHM)
print('\n')

# plt.figure(1, figsize=(10, 15))
plt.plot(xdata, ydata, 'ko', label='data')
plt.plot(xdata, gauss(xdata, *gauss_fit(xdata, ydata)), '--r', label='fit')
plt.legend()
plt.title('Gaussian fit,  $f(x) = A e^{(-(x-x_0)^2/(2sigma^2))}$')
plt.xlabel('Bins')
plt.ylabel('n')
plt.savefig('diff_x.pdf')
plt.show()

xdata = bins_y
ydata = n_y

H, A, x0, sigma = gauss_fit(xdata, ydata)
FWHM = 2.35482 * sigma

print('The offset of the gaussian baseline is', H)
print('The center of the gaussian fit is', x0)
print('The sigma of the gaussian fit is', sigma)
print('The maximum intensity of the gaussian fit is', H + A)
print('The Amplitude of the gaussian fit is', A)
print('The FWHM of the gaussian fit is', FWHM)
print('\n')

# plt.figure(1, figsize=(10, 15))
plt.plot(xdata, ydata, 'ko', label='data')
plt.plot(xdata, gauss(xdata, *gauss_fit(xdata, ydata)), '--r', label='fit')
plt.xlabel('Bins')
plt.ylabel('n')
plt.savefig('diff_y.pdf')
plt.show()

xdata = bins_z
ydata = n_z

H, A, x0, sigma = gauss_fit(xdata, ydata)
FWHM = 2.35482 * sigma

print('The offset of the gaussian baseline is', H)
print('The center of the gaussian fit is', x0)
print('The sigma of the gaussian fit is', sigma)
print('The maximum intensity of the gaussian fit is', H + A)
print('The Amplitude of the gaussian fit is', A)
print('The FWHM of the gaussian fit is', FWHM)
print('\n')

# plt.figure(1, figsize=(10, 15))
plt.plot(xdata, ydata, 'ko', label='data')
plt.plot(xdata, gauss(xdata, *gauss_fit(xdata, ydata)), '--r', label='fit')
plt.xlabel('Bins')
plt.ylabel('n')
plt.savefig('diff_z.pdf')
plt.show()
