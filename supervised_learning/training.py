from sklearn import naive_bayes
from sklearn import datasets
from collections import Counter

nb = naive_bayes.MultinomialNB(fit_prior=True)  # un algo d'apprentissage

irisData = datasets.load_iris()
# Training 
nb.fit(irisData.data[:-1], irisData.target[:-1]) 
# Test
p31 = nb.predict([irisData.data[31]])  
print(p31)
plast = nb.predict([irisData.data[-1]]) #  test on the last one
print(plast)
# Prediction on the entire dataset
p = nb.predict(irisData.data[:])
print(p)
print(Counter(p))