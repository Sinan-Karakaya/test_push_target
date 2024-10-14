This is a technique that consists in creating synthetic samples for a minor class, to reduce the gap in amount of data between classes.

`I have a 1000 data sample for my class A, but only 400 for my class B, I need to make them even!`

The ***SMOTE*** helps to avoid overfitting issues that can happen with random oversampling.

The ***SMOTE*** can be used for binary classification or multiclass, et use different parameters to control the sampling strategy.

## Pros

1. Helps to balance the difference in the amount of data for classes 
2. Doesn't lose informations unlike *subsampling*, which cuts data points from the "biggest" class.
3. Helps to avoid overfitting.
4. Easy to implement with libraries such as `imbalanced-learn` in Python

## Cons

1. Can increase the time to train the model and its complexity, since it creates new data
2. Can introduce noise or outliers in the dataset if the samples are not representative of the minor class and/or it is too close to the frontier with the major class
3. Is not made for *categorical variables*, since it uses the Euclidian distance to determine the neighbors. In this case, the [[SMOTE-NC]] should be used.