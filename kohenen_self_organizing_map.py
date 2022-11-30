## The following code uses self organizing map (or often called self organizing feature maps or Kohenian self organinzing feature maps or KSOFM) to generate features or//
## centroids from the input data from the MNIST digit based dataset. I have used torchvision MNIST dataset as an input images of size 28x28 and user defined function//
## to model the netwrok for KSOFM. The complete code is based on python3 and anaconda was used as the compiler environment

# importing the necessary modules
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision

## The number of iterations or epochs to train the data
num_epochs = 15

class KSOFM():
    ## Decalring the variables and values associated
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1):
        self.m = m
        self.n = n
        self.dim = dim
        self.shape = (m, n)
        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.half_size = 31
        rng = np.random.default_rng(seed=61)
        self.weights = rng.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)
        self._trained = False
    ## 
    def _get_locations(self, m, n):
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)
    ## Finding the best matching units or BMU based on the eucleadian distance to get the weights that are closest to the training examples
    def _find_bmu(self, x):
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        distance = np.linalg.norm(x_stack - self.weights, axis=1)
        return np.argmin(distance)
    ## performing exponential neighborhood decay and updaitng the weights with bmu
    def step(self, x, ti, gi):
        decay_iter = int(ti / 31)
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        bmu_index = self._find_bmu(x)
        bmu_location = self._locations[bmu_index,:]
        start_index = bmu_location[1] - self.half_size
        if start_index < 0: start_index = 0
        end_index = bmu_location[1] + self.half_size
        if end_index > 40: end_index = 40
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)
        neighborhood = np.exp((bmu_distance / 2*(self.sigma ** 2)) * - 1)
        if (gi != 0 and gi % decay_iter ==0):
            self.half_size = self.half_size - 1
            print("The shriked window: ", self.half_size)
        neighborhood[:start_index] = 0
        neighborhood[end_index+1:40] = 0
        local_step = self.lr * neighborhood
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)
        delta = local_multiplier * (x_stack - self.weights)
        self.weights += delta
        
    ## fitting the data for each iteration
    def fit(self, X, epochs=1, shuffle=True):
        global_iter_counter = 0
        n_samples = X.shape[0]
        total_iterations = epochs * 60000
        for epoch in range(epochs):
            if shuffle:
                rng = np.random.default_rng(seed=61)
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)
            for idx in indices:
                global_iter_counter += 1
                input = X[idx]
                self.step(input, total_iterations, global_iter_counter)
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
        self._trained = True
        return
    
    def transform(self, X):
        X_stack = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)
        diff = X_stack - cluster_stack
        return np.linalg.norm(diff, axis=2)
    @property
    def cluster_centers_(self):
        return self.weights.reshape(self.m, self.n, self.dim)
    ## training the input dataset
    def Train():
        mean = (0.1307, )
        std = (0.3081, )
        train_trans = transforms.Compose([transforms.RandomRotation((0, 5), fill=(0, )),])
        dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
        images = []
        for img, _ in dataset:
            img = (np.asarray(train_trans(img)) - mean) / std
            images.append(img)
        images = np.array(images)
        images = images.reshape(images.shape[0], 784)
        model = SOM(m=1, n=40, dim=784)
        model.fit(images, epochs=num_epochs)
        centers = model.cluster_centers_.squeeze(axis=0)
        iter = 0
        for center in centers:
            iter += 1
            plt.imshow(center.reshape(28,28), cmap='pink')
    Train()
