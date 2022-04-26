import random
from sklearn import datasets
from collections import Counter
import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split


def split(S):
    data = S.data[:]
    target = S.target[:]
    dataS1 = np.array([[]])
    targetS1 = np.array([])
    dataS2 = np.array([[0, 0, 0, 0]])
    targetS2 = []
    n = len(target)
    print(Counter(target))
    for i in range(n//3):
        randomIndex = random.randrange(len(target))
        dataS2 = np.append(dataS2, [data[randomIndex]], axis=0)
        print(target[randomIndex])
        targetS2.append(target[randomIndex])
        data = np.delete(data, randomIndex, 0)
        # print(target)
        target = np.delete(target, randomIndex, 0)
        dataS1 = data[:]
        targetS1 = target[:]
    dataS2 = np.delete(dataS2, 0, 0)
    # targetS2 = np.array(targetS2)
    print(Counter(targetS1))
    print(Counter(targetS2))

    return [dataS1, targetS1, dataS2, targetS2]


irisData = datasets.load_iris()
S = split(irisData)
S1 = train_test_split(irisData.data, irisData.target, test_size=0.33)


def test(S, clf):
    clf.fit(S[0][:], S[1][:])
    p = clf.predict(S[2][:])
    return 1-clf.score(S[2][:], S[3][:])


def test2(S, clf):
    clf.fit(S[0][:], S[2][:])
    p = clf.predict(S[1][:])
    return 1-clf.score(S[1][:], S[3][:])


nb = naive_bayes.MultinomialNB(fit_prior=True)
#print(Counter(test(S, nb)))


def repeatTest(t, S, clf):
    result = 0
    for i in range(t):
        result += test2(S, clf)
    return result/t


# result = 0
# for i in range(20):
#     result += repeatTest(1000, S, nb)
#     result = result/20
# print(result)
# print(repeatTest(1000, S, nb))
print(repeatTest(20000, S1, nb))
