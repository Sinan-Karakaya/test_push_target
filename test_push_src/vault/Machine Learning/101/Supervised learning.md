# Types of output

- Classification (multi class or binary)
- Regression (prediction of continuous values)

# Example of a supervised learning

| Feature 1 | Feature 2 | Feature 3 | Label |
| --------- | --------- | --------- | ----- |
| 6 | 1 | 2 | 1 |
| 1 | 0 | 14 | 0 |
| 2 | 3 | 5 | 1 |

A dataset is composed of 2 groups: **features** and **labels**. The set of features would be called a **feature matrix**, a row of features would be called a **feature vector,** and the label for that row would be the **target for that feature vector**. 

## In short, the training

Features are considered the inputs of our model, they will be computed in some form and give an output.

Labels are in comparison the output of our model you could say. In binary models, they are either 0 or 1, whereas in multi class models, this would be different.

In a supervised learning, labels are already given to the models for its training. This way, we, and the model, know what result to expect. This will help automatically validate the guess of the model during its training.

#### Divide to reign

How do we ensure that our model can work with never seen before data ? We break our dataset in 3 categories:

- The training dataset
- The validation dataset 
- The testing dataset

Once our model gives us a vector of prediction, we will compare it with the expected vector, and the difference (called the **loss**) will be used to make adjustments to our training. 

The validation set will be used as a reality check during/after the training to ensure our model can handle unseen data.

### Let's talk about the Loss

The higher the loss is, the less good is the result. So, the closer we get to 0 the better. Once the best performing model has been chosen, it will be testing time.

This is where our __testing dataset__ comes in. It will be used to check how generalizable the final chosen model is. The performance of the model on that dataset is what will be the final reported performance. Basically:

``Yeah, this is how the AI will perform on any type of data``

Now, how do we actually calculate our loss ?

L1 loss:
$$ loss = sum(| y_{real} - y_{predicted} |) $$

L2 loss (the closer the less bad the penalty, and vice-versa):
$$ loss = sum((y_{real} - y_{predicted}) ^ 2) $$

Binary Cross-Entropy Loss (the loss decreases as the performance gets better):
$$ loss = -1/N*sum(y_{real}*log(y_{predicted}+(1-y_{real})*log((1-y_{predicted}))) $$

#### Metrics of performance 

The main metrics of performance for a model would be its accuracy, usually in %.