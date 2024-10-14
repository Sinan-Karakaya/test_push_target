## Decision Tree

This is a graphical representation of a series of decisions and their possible outcomes. It is a hierarchical model that splits the data into smaller subsets based on some criteria, such as a feature value or a threshold. Each split creates a branch in the tree and each branch ends with a leaf node that represents a class label (for classification) or a numerical value (for regression).

This can also be used to visualize the logic behind a prediction by showing the rules and conditions that led to it.

![[Pasted image 20231123220042.png]]

## Random Forest

A *Random Forest* is just a set of *Decision Tree* that were all trained on a different set of data, and a different set of features. The final prediction is obtained by averaging the predictions of all the tress for regression tasks, or by taking the majority vote for classification.

This is an example of a ***Ensemble Learning Model***, which aims to improve the accuracy and robustness of a single model by combining several models.

### Pros

1. Can handle both classification and regression 
2. Can deal with missing values and outliers
3. Can reduce the risk of overfitting by averaging the predictions of many trees
4. Can capture complex and non-linear relationships among the features 

### Cons

1. Requires more computational power and resources
2. Takes longer to train and to make predictions
3. Is less interpretable and explainable than a single tree
4. Might not perform well on very high dimensional or sparse data
