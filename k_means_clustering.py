import numpy as np
import sklearn.datasets, sklearn.decomposition
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

(x_train, y_train), ((x_test, y_test)) = tf.keras.datasets.mnist.load_data()
x = x_train.reshape(-1,28*28)

print(x_train.shape)

# test_case
digit = x[32]
digit.reshape(28,28)
#plt.title('test_image')

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans

k_means_5cent = KMeans(n_clusters = 5, max_iter = 1000).fit(x)
k_means_centroid_5 = k_means_5cent.cluster_centers_
k_means_centroid_5.shape

axes = []
fig = plt.figure(1, figsize=(10,10))

for i in range(5):
    axes.append(fig.add_subplot(1,5, i+1) )
    axes[i].set_title("Centroid" + str(i+1))
    plt.imshow(k_means_centroid_5[i].reshape(28,28), cmap='pink')
fig.tight_layout()
plt.show()
