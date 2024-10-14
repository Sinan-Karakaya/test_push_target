![[Pasted image 20231120224000.png]]

We can change the probability to be an odd/ratio, which is allowed to have a infinite amount of values.
$$ \frac{p}{1 - p} = mx+b $$
The issue is that mx+b can be negative, which we don't want. We can fix that by taking the log of the odds.
$$ ln(\frac{p}{1 - p}) = mx+b $$
Now we can solve the probability:
$$  e^{ln(\frac{p}{1 - p})} = e^{mx+b} $$
$$  \frac{p}{1 - p} = e^{mx+b} $$
$$ p = (1-p) * e^{mx+b} $$
$$ p = e^{mx+b} - pe^{mx+b} $$
$$ p(1 + e^{mx+b}) = e^{mx+b} $$
$$  p = \frac{e^{mx+b}}{1+e^{mx+b}} $$
$$   p = \frac{e^{mx+b}}{1+e^{mx+b}} * \frac{e^{-(mx+b)}}{e^{-(mx+b)}} $$
$$    p = \frac{1}{1+e^{-(mx+b)}} $$
The last step let us transform our formula into a form called a **sigmoid function**, which look like this:
$$ S(x) = \frac{1}{1+e^{-x}} $$
![[Pasted image 20231120225039.png]]

This shape would fit perfectly for our afore mentioned case, almost giving us a binary function.

If we have a single feature, this would be a **simple Logistic Regression**.
Whereas if we have multiple features, this would be a **multiple Logistic Regression**.

#### Implementation

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)
print()
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support  

#            0       0.68      0.72      0.70      1320
#            1       0.85      0.82      0.83      2484

#     accuracy                           0.79      3804
#    macro avg       0.76      0.77      0.77      3804
# weighted avg       0.79      0.79      0.79      3804
```