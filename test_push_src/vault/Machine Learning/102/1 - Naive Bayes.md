# What are they ?

| Has covid ? | + | - | Total |
| - | - | - | - |
| y | 531 | 6 (false negative) | 537 |
| n | 20 (false positive) | 9443 | 9463 |
| Total | 551 | 9449 | |

*A false positive can be known as a Type I error.*
*A false negative can be known as a Type II error.*

#### What is the probability of having covid given a positive test ?

This corresponds to the first row of the dataset.
$$ P(covid /+ test) = 531 / 551 = 96.4\% $$
## Bayes' Rule

`What is the probability of event A given that event B happen ?`
$$ P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$$
P(false positive) = 0.05
P(false negative) = 0.01
P(disease) = 0.1

$$ P(disease|(+)test) = \frac{P((+)|disease) * P(disease)}{P(+)} $$
$$  P(disease|(+)test) = \frac{0.99 * 0.1}{0.99 * 0.1 + 0.05 * 0.9} $$
$$ P(disease|(+)test) = 0.6875 $$

We can apply Bayes' Rule to classification.

#### Some terminology

![[Pasted image 20231120220454.png]]

- **Posterior**: What's the probability of some class C<sub>k</sub> ? What's the probability of having this feature vector X fitting into C<sub>k</sub> ?
- **Likelihood**: What's the likelihood of seeing X inside C<sub>k</sub> ?
- **Prior**: What is the probability of this class in general in the population ?
- **Evidence**: We're creating the posterior probability built upon the prior using some sort of evidence.

#### Derivation

$$ P(C_k | x_1, x_2, ..., x_n) \propto p(C_k) \prod_{i=1}^{n} p(x_i | C_k) $$

The first part asks what are the condition that the condition is 'true' depending on those features. (If it's rainy and windy and dark, should I play outside ?)

Using Bayes' Rule, we get:
$$ P(C_k | x_1, x_2, ..., x_n) = \frac{P(x_1, x_2, ..., x_n | C_k) * P(C_k)}{P(x_1, x_2, ..., x_n)} $$
The denominator (bottom part) has no incident on the result, so we can make it a constant and remove it.
$$ P(C_k | x_1, x_2, ..., x_n) \propto P(x_1, x_2, ..., x_n | C_k) * P(C_k) $$
In our example, we assume that our features are independents from each other, which let us expand our long probability in the second part to this:
$$ P(x_1, x_2, ..., x_n | C_k) = P(x_1 | C_k) * P(x_2 | C_k) * ... * P(x_n | C_k) $$
So now, we can write it like this:
$$ \propto P(C_k) \prod_{i=1}^n P(x_i | C_k) $$
Basically, the probability that you know that we're in some category, given that we have all these different features is proportional to the probability of that class in general, times the probability of each of those features, given that we're in this one class that we're testing.

So the probability us playing soccer today given that it's rainy, not windy, and it's Wednesday, is proportional to `what is the probability that we play soccer anyway, and then times the probability that it's rainy, given that we're playing soccer, etc...`

#### Use this to make our classification 

$$ \hat{y} = argmax_{k \varepsilon {1, k}} P(C_k | x_1, x_2, ..., x_n) $$
Thanks to Bayes' Rule, we can simplify to:
$$ \hat{y} = P(C_k) \prod_{i=1}^n P(x_i | C_k) $$
This action is known as MAP (Maximum A Posteriori). `Pick the K that is the most probable so we minimize the probability of misclassification.`

#### Implementation

```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)
print()
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support

#            0       0.71      0.41      0.52      1334
#            1       0.74      0.91      0.82      2470

#     accuracy                           0.73      3804
#    macro avg       0.72      0.66      0.67      3804
# weighted avg       0.73      0.73      0.71      3804
```