# ML-Labs

## Supervised Learning 
In this lab we used the IRIS dataset: 

- Data visiualisation to understand our dataset using pylab.scatter
<img src="https://github.com/rihemebh/ML-Labs/blob/main/scatter.PNG" />

- Trainig and prediction using Naive bayes

- Evaluate the classifier performance using 3 different methods 
  - [Sum function](https://github.com/rihemebh/ML-Labs/blob/main/supervised_learning/error.py)
  - [the difference between arrays](https://github.com/rihemebh/ML-Labs/blob/main/supervised_learning/error_diff.py)
  - [Accuracy](https://github.com/rihemebh/ML-Labs/blob/main/supervised_learning/error_score.py)


## Unsupervised learning 

In this lab we used a dataset of cheese in order to classify them into different types based on their composition.

for that we used 4 algorithms

- k-means 
- Hierarchical ascending classification 
- Agglomerative Clustering
- Diana


### Implementation of Divisive clustering (DIANA) with k-means 
Algorithm : 

Apply kmeans algorithm on the whole dataset with number of clusters = 2, in each resulted cluster we re-apply kmeans recursively until we get a cluster of one element.
Every time we have to put the linkage data in Z matrix in order to generate the dendrogram 

- Linkage data = 
[
index of element 1, 
index of element 2, 
distance between element1 and 2, 
clusters length
]

