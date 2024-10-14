False positives and false negatives are errors of classification that can happen. Their importance are tied to the context and the consequences of the output of the model.

There is a compromise between false negatives and false positives that can be visualized with a ***Receiver Operating Characteristics (ROC)*** graph.

## What is it

This shows the relation between the rate of true positives (*sensibility*) and the rate of false positives (1-specified) pour different degree of decision. The choice of the optimal degree depends on the relative cost of those errors of classification and the distribution of the classes.

![[Pasted image 20231123213150.png]]
## How to reduce the amount of false positives and false negatives 

There are multiple possible strategies, such as:

- Improve the quantity and quality of the training dataset, and avoid bias, missing values and outliers.
- Choose the most fitted model for the problem, by taking into account the complexity, and other parameters.
- Use techniques such as *bagging*, *boosting* or *stacking*, to combine the predictions of multiple models et reduce the variance.
- Use methods of resampling, like [[SMOTE (Synthetic Minority Oversampling Technique)]], to balance the unbalanced classes and avoid overfitting