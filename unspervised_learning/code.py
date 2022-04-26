import pandas as pd
import pandas
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Reading the dataset
fromage = pd.read_table(r"fromage1.txt", sep="\t", header=0, index_col=0)
#pd.plotting.scatter_matrix(fromage, figsize=(9, 9))

np.random.seed(0)
# The number of cluster resulted is 4
kmeans = cluster.KMeans(n_clusters=4)
# Training
kmeans.fit(fromage)

# generate an array of indexes of sorted  data
idk = np.argsort(kmeans.labels_)

# labels_ : les numéros de cluster de chaque observation
print(pd.DataFrame(fromage.index[idk],kmeans.labels_[idk]))

# Calculate the distance of elements
print(kmeans.transform(fromage))

# librairies pour la CAH

Z = linkage(fromage, method='ward', metric='euclidean')
# affichage du dendrogramme

plt.title('CAH avec matérialisation des 4 classes')
dendrogram(Z, labels=fromage.index, orientation='left', color_threshold=255)
plt.show()
groupes_cah = fcluster(Z, t=255, criterion='distance')
print(groupes_cah)

# index triés des groupes
idg = np.argsort(groupes_cah)
# affichage des observations et leurs groupes

print(pandas.DataFrame(fromage.index[idg], groupes_cah[idg]))
print(pandas.crosstab(groupes_cah, kmeans.labels_))
# l’analyse en composantes principales
acp = PCA(n_components=2).fit_transform(fromage)
for couleur, k in zip(['red', 'blue', 'lawngreen', 'aqua'], [0, 1, 2, 3]):
    plt.scatter(acp[kmeans.labels_ == k, 0], acp[kmeans.labels_ == k, 1], c=couleur)
#plt.show()
