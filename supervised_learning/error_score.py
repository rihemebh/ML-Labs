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

print((1-nb.score(irisData.data, irisData.target))*100,'%')
