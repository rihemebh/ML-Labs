import matplotlib
from itertools import cycle
import pylab as pl
from sklearn import datasets
irisData = datasets.load_iris()


def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')  # cycle de couleurs
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        print(i)
        print(data[target == 1])
        pl.scatter(data[target == i, 2],
                   data[target == i, 3], c=c, label=label)
    # droite qui sépare deux classes
    pl.plot([2.5, 2.5], [0, 2.5], linestyle='dotted')
    pl.legend()  # afficher la légende
    pl.show()



plot_2D(irisData.data, irisData.target, irisData.target_names)
