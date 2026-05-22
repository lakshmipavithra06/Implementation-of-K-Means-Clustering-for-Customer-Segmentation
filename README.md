# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the customer dataset using Pandas.
2. Select Annual Income and Spending Score columns as input features for clustering.
3. Apply the K-Means algorithm with 5 clusters and predict customer groups.
4. Plot the clustered customers and cluster centroids using Matplotlib.
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: LAKSHMI PAVITHRA M
RegisterNumber:212225220055
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customer.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X')

# Labels
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()

```

## Output:
<img width="1330" height="585" alt="image" src="https://github.com/user-attachments/assets/e72f112a-d5de-4aba-bfc6-df0eb5d65d30" />

<img width="921" height="578" alt="image" src="https://github.com/user-attachments/assets/093d6102-8548-46c8-8c9c-9214ac691ffd" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
