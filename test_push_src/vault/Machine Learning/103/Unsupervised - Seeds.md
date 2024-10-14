We are going to implement our first [[Unsupervised learning]] model, using this dataset:

[seeds - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds)

## Implementation

### Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']
df = pd.read_csv('sample_data/seeds_dataset.txt', names=cols, sep='\s+')

for i in range(len(cols) - 1):
    for j in range(i + 1, len(cols)):
        x_label = cols[i]
        y_label = cols[j]
        sns.scatterplot(x=x_label, y=y_label, hue='class', data=df)
        plt.show()
```

First, we gather our data, and display it firstly by groups.

![[Pasted image 20231122163143.png]]

![[Pasted image 20231122163201.png]]

![[Pasted image 20231122163211.png]]

We can see distinct groups in the data.

### [[K-MEANS Clustering]] 

#### Implementation

```python
from sklearn.cluster import KMeans

a = 'perimeter'
b = 'asymmetry'
X = df[[a, b]].values

kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.labels_
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[a, b, 'class'])

# K means clustering (Fig.1)
sns.scatterplot(x=a, y=b, hue='class', data=cluster_df)
plt.show()
```

![[Pasted image 20231122164123.png]]

### Higher dimensions

```python
X = df[cols[:-1]].values
kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.labels_

cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=df.columns)

sns.scatterplot(x=a, y=b, hue='class', data=cluster_df)
plt.show()
```

![[Pasted image 20231122164750.png]]

### PCA

