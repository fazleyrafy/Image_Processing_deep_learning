/*
*
* @author MFRafy
*
* created on 13 September, 2022
* The 10-class digits in MNIST training-dataset has been used to perform PCA using python 
* In the following you will find the source code to Display all 30 eigenvectors as images and to plot  top 30 eigenvlaues
* Defined funtions that takes any digits (single digit) and number of PCA eigen vectors to compare the original and reconstructed images of the digit with the
* Mean Squared Error (MSE) of the corresponding digit
*
*/


## importing all the necessary modules

import numpy as np
import sklearn.datasets, sklearn.decomposition
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler


## load the mnist dataset and assign tringing and test data
(x_train, y_train), ((x_test, y_test)) = tf.keras.datasets.mnist.load_data()

## Data Preprocessing
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

## Before applying PCA, we need to convert mean=0 and standard deviation=1 for each variable

x = x_train.reshape(-1,28*28)
x = x/255.0
mean = np.mean(x,axis=0)
x_std = x - mean
covariance = np.cov(x_std.T)

## Compute eigenvalues and eigenvectors for different eigen values and corresponding eigen vectors (considering the largest ones)

eig_val30, eig_vec30 = eigh(covariance,eigvals = (783-29,783))
eig_val20, eig_vec20 = eigh(covariance,eigvals = (783-19,783))
eig_val5, eig_vec5 = eigh(covariance,eigvals = (783-4,783))
eig_val2, eig_vec2 = eigh(covariance,eigvals = (782,783))

## ploting only the top 30 eigen vectors
rows, cols = 6, 5
axes=[]
fig=plt.figure(1,figsize=(10,10))
for i in range(rows*cols):
    v = eig_vec30[:,-(i+1)]
    axes.append(fig.add_subplot(rows, cols, i+1) )
    subplot_title=("Vector"+str(i+1))
    axes[-1].set_title(subplot_title)
    plt.imshow(v.reshape(28,28), cmap='pink')
fig.tight_layout()
plt.show()

## plotting the top 30 eigenvalues

x_ax = np.arange(0,30)

plt.plot(x_ax, eig_val30, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=5)
plt.title("Top 30 Eigenvalues")
plt.xlabel("index")
plt.ylabel("EigenValues")
plt.show()

## Projecting the original data sample on the plane formed by 30, 20, 5, and two principal eigenvectors by vector-vector multiplication

vectors30 = eig_vec30.T
new_coordinates20 = np.matmul(vectors20, x_std.T)
vectors20 = eig_vec20.T
new_coordinates10 = np.matmul(vectors10, x_std.T)
vectors5 = eig_vec5.T
new_coordinates5 = np.matmul(vectors5, x_std.T)
vectors2 = eig_vec2.T
new_coordinates2 = np.matmul(vectors2, x_std.T)

## importing training data of MNIST dataset

df = pd.read_csv('train.csv')
new_df = df.drop('label', axis=1)
recon_df = new_df
labels = df['label']


## function returning original images of number #

def org_image(digit):
    x2 = new_df.loc[digit-1].values.reshape(28,28)
    plt.imshow(x2, cmap='pink')
    
## showing original iamge of digit #4
org_image(4)


## function returning Mean Squared Error (MSE) for digit # for # PCA component

def MSE_error(digit, pca_com):
    x2 = new_df.loc[digit-1].values.reshape(28,28)
    x3 = standardized_scalar.fit_transform(x2)
    #print(x3.shape)
    pca = PCA(n_components = pca_com)
    X_re = pca.fit_transform(x3)
    X_rec = pca.inverse_transform(X_re)
    # print(X_rec.shape)
    print(f'MSE error for PCA: {sklearn.metrics.mean_squared_error(x3,X_rec)}')

## example case for digit "4" with "10" PCA

MSE_error(7,10)

## function returning reconstructed images of number # with # of PCA components

def rec_image(digit, pca_com):
    x2 = new_df.loc[digit-1].values.reshape(28,28)
    x3 = standardized_scalar.fit_transform(x2)
    #print(x3.shape)
    pca = PCA(n_components = pca_com)
    X_re = pca.fit_transform(x3)
    X_rec = pca.inverse_transform(X_re)
    
    plt.title("Reconstructed Image of #%i" %digit)
    plt.imshow(X_rec, cmap='pink')
## example case for digit "4" with "10" PCA
rec_image(4, 10)
