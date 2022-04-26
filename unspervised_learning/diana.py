import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn import cluster

"""
we need this function to calculate the distance between 2 elements
so if we already calculate the mean value we get it from the Z matrix
or we calculate it : sum(fromage.iloc[i])/9
"""


def distance(i):
    if i >= len(fromage):
        return Z[i - len(fromage)][2]
    else:
        return sum(fromage.iloc[i]) / 9


"""
the length of cluster 
we need it for the linkage matrix
"""


def cluster_len(i):
    if i >= len(fromage):
        return Z[i - len(fromage)][3]
    else:
        return 1


"""
we apply kmeans on our dataset then we split it into 2 clusters a0 and a1 
"""


def kmeans_split(c):
    np.random.seed(0)
    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(c)

    idk = np.argsort(kmeans.labels_)
    idk0 = []
    idk1 = []

    k = 0
    for i in kmeans.labels_[idk]:
        if i == 0:
            idk0.append(idk[k])
            k = k + 1
        else:
            idk1.append(idk[k])
            k = k + 1

    a0 = pd.DataFrame(c.values[idk0], c.index[idk0])
    a1 = pd.DataFrame(c.values[idk1], c.index[idk1])
    return a0, a1


"""
The idea is to apply kmeans on the whole dataset 
we get 2 clusters and in each cluster w re-apply kmeans recursively until we get a cluster of one element
in everytime we have to to put the linkage data in Z matrix in order to generate the dendrogram

"""


def diana(c):
    if len(c) < 2:
        ind = np.argwhere(fromage.index == c.index[0])
        return ind[0][0]
    a0, a1 = kmeans_split(c)
    clusters.append(a0)
    clusters.append(a1)
    index1 = diana(a0)
    index2 = diana(a1)
    Z.append([
        index1,
        index2,
        distance(index1) + distance(index2),
        cluster_len(index1) + cluster_len(index2)
    ])

    return len(Z) - 1 + len(fromage)


fromage = pd.read_table(r"fromage1.txt", sep="\t", header=0, index_col=0)

Z = []
clusters = []
index = diana(fromage)

fig = plt.figure(figsize=(25, 10), dpi=100)
plt.title('DIANA')
dn = dendrogram(Z, labels=fromage.index, orientation='top', color_threshold=1500)
plt.show()
