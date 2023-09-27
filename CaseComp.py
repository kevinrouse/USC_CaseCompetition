#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:26:00 2022

@author: kevinrouse78
"""
import pandas as pd
import numpy as np
import os
#load data 
os.getcwd()
os.chdir('/Users/kevinrouse78/Desktop')
d = pd.read_csv("RNAseq.csv")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
print(d.isna().sum())
#No NA values
#z-score normalization
labels = d.iloc[:,1]
features = d.iloc[:,2:]
row_means = features.mean(axis =1)
row_stds = features.std(axis=1)
features = (features.T - row_means) / row_stds
index = list(features.index.values)
features.columns = labels


pca = PCA(n_components=2)
pca_data = pca.fit_transform(features)


# Choose K-means as the clustering algorithm
kmeans = KMeans(n_clusters=2)

# Fit the K-means model to the data
kmeans.fit(pca_data)

# Predict the group membership for each sample
k_labels = kmeans.labels_


groups = pd.DataFrame(k_labels, index=index)

Ones = groups.drop(groups[groups[0]==0].index, axis=0)
Zeros = groups.drop(groups[groups[0]==1].index, axis=0)

import matplotlib.pyplot as plt

# Plot the first two principal components
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=k_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


