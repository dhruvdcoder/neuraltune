from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

X,_ = load_diabetes(return_X_y=True)
print(X.shape)

def kmeans(X):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 11)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = kmeanModel.inertia_
        mapping2[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                     'euclidean'),axis=1)) / X.shape[0]

    plt.figure(1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.savefig("Figures/elbow.jpg")

    plt.figure(2)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig("Figures/elbow1.jpg")

    kmeans_ = KMeans(n_clusters=2, random_state=0).fit(X)
    print(kmeans_.labels_)
    print(kmeans_.cluster_centers_)
    return kmeans_.cluster_centers_

if __name__ == '__main__':
    output = kmeans(X)