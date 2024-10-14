# Types of ML

- [[Supervised learning]]
- [[Unsupervised learning]]
- Reinforcement learning

## The project

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The cols of the csv/labels, are not included in the sample data
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv('sample_data/magic04.data', names=cols)

# Labels are g or h, we need to convert them to 0 or 1
df["class"] = (df["class"] == "g").astype(int)
```

#### Show our data

```python
for label in cols[:-1]:
    # Everywhere the label == 1, we get the label and draw it on the histogram
    plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)

  

    # The opposite
    plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)

    plt.title(label)
    plt.ylabel("Probability") # Since its a density plot
    plt.xlabel(label)
    plt.legend()
    plt.show()
```

#### Train, validation, test datasets

```python
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# 0-60% train, 60-80% validation, 80-100% test
# The sample call shuffle the data
train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

def scale_dataset(df, oversample = False):
    # We assume that the last column is the label
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # There are more gamma than hadron events, so we need to balance the dataset
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    # horizontally stack the features and labels
    # The reshape makes it a column vector
    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

  

# We scale the data and oversample the train dataset (len of gamma and hadron events are the same)
train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)
```

#### KNN: K Nearest Neighbors

Let's take this graph for example:

![[Pasted image 20231120212029.png]]

We can deduce that this is binary classification, since each sample has a label.

![[Pasted image 20231120212356.png]]

Here, we added a point. From this point, we calculated the **Euclidean distance** from every other points, using this formula:
$$ \sqrt{(x_1-x_2)^2+(y_1-y_2)^2} $$
The K in KNN stands for `How many neighbors do we use in order to judge what the label is ?`. Let's take k=3.

![[Pasted image 20231120212852.png]]

All the neighbors are blue, so chances are that the point we choose would also be blue.

![[Pasted image 20231120213003.png]]

Same story here.
If all the nearest neighbors aren't the same, we would take the majority.

The interesting part is that in case we have more features, we can integrate those in the formula above, so that it counts all those dimensions, letting us figure out which point is the closest to the point we desire to classify.

#### Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support
#            0       0.75      0.73      0.74      1394
#            1       0.85      0.86      0.85      2410

#     accuracy                           0.81      3804
#    macro avg       0.80      0.80      0.80      3804
# weighted avg       0.81      0.81      0.81      3804
```