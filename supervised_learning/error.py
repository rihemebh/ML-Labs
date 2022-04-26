from sklearn import datasets
from sklearn import naive_bayes
from collections import Counter

irisData = datasets.load_iris()


irisData = datasets.load_iris()
nb = naive_bayes.MultinomialNB(fit_prior=True)
# Training
nb.fit(irisData.data[:], irisData.target[:])
# Test
p = nb.predict(irisData.data[:])

y = irisData.target

print(Counter(y))
print(Counter(p))
ea = 0
for i in range(len(irisData.data)):
    if p[i] != y[i]:
        ea = ea+1
print((ea/len(irisData.data))*100, "%")
