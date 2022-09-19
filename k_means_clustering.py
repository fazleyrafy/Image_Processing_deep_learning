## importing all the necessary modules

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans

## loading the MNIST dataset and reshaping the training dataset for further experiment
(x_train, y_train), ((x_test, y_test)) = tf.keras.datasets.mnist.load_data()
x = x_train.reshape(-1,28*28)

## function to fit the data of x with k_means of # of clusters and setting centroids
# here n is the number of centroids in n number of clusters
def apply_kmeans(n):
    k_means_mnist = KMeans(n_clusters = n, max_iter = 1000).fit(x)
    k_means_centroid_n = k_means_mnist.cluster_centers_
    return k_means_centroid_n

## function to display the images of # of centroids
def display_kmeans(n, arr):
    axes = []
    fig = plt.figure(2, figsize=(12,12))
    for i in range(n):
        if(n<=2):    
            axes.append(fig.add_subplot(1,2, i+1))
        elif(n>2):
            axes.append(fig.add_subplot(5,6, i+1))
        axes[i].set_title("Centroid" + str(i+1))
        plt.imshow(arr[i].reshape(28,28), cmap='pink')

## results for only 30 centroids with images

temp = apply_kmeans(30)
display_kmeans(30, temp)
