from sklearn import datasets
from collections import Counter
irisData = datasets.load_iris()
print(irisData.data)
print(irisData.target)
print(irisData.target_names)
print(irisData.feature_names)
print(irisData.data.shape)  # (150,4) 150 instances and 4 features
print("___________________________________________________________")
# 2 question

# Combien y a-t-il de données dans chaque classe

occurences= Counter(irisData.target)
for i in [0,1,2]:
    print(f"La classe {i}: {irisData.target_names[i]} a {occurences[i]} instances" )
# les attributs et la classe du 32ème élément de l'échantillon
print(irisData.data[31])
print(irisData.target[31])

