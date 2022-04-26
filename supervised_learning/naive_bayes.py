from sklearn import naive_bayes
from sklearn import datasets
irisData = datasets.load_iris()

nb = naive_bayes.MultinomialNB(fit_prior=True)
nb.fit(irisData.data[:99], irisData.target[:99])
print(nb.predict(irisData.data[100:149]))
# on doit faire le shuffle car les derniers 50 sont dans la meme classe