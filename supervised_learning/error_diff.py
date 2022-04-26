from sklearn import datasets
from sklearn import naive_bayes
from collections import Counter
import numpy
irisData = datasets.load_iris()


irisData = datasets.load_iris()
nb = naive_bayes.MultinomialNB(fit_prior=True)
# Training
nb.fit(irisData.data[:], irisData.target[:])
# Test
p = nb.predict(irisData.data[:])

y = irisData.target

diff = p-y
print(numpy.nonzero(diff)) # afficher les indices des valeurs diff de 0
print(diff)
print(numpy.count_nonzero(p-y)*100/len(irisData.target), "%")
