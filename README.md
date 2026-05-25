# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select relevant features.

2.Standardize the features for better clustering.

3.Apply K-Means with a chosen number of clusters (k).

4.Predict cluster labels for each customer.

5.Visualize and interpret the customer segments. 
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SUDHARSAN U
RegisterNumber: 212225040433
*/
```
```python
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print("Dataset loaded successfully!")
print(df.head())

# Step 3: Select Relevant Features (Example: 'Annual Income' and 'Spending Score')
X = df.iloc[:, [3, 4]].values  # Adjust column indices as needed

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Find the optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Step 6: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 7: Visualize the clusters
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Feature 1 (e.g., Annual Income)')
plt.ylabel('Feature 2 (e.g., Spending Score)')
plt.legend()
plt.show()
```

## Output:
<img width="821" height="728" alt="image" src="https://github.com/user-attachments/assets/5e0e5934-be15-4298-8a4b-0b248bb77f2c" />

<img width="808" height="577" alt="image" src="https://github.com/user-attachments/assets/12475074-e96a-44a6-8a14-75ca1e340e88" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
