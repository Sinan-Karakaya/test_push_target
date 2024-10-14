![[Pasted image 20231121103150.png]]

In our case here, we want to divide the different labels in group. In our case it would be represented as a line since it only has 2 features, but it would become a plane in 3D if we had more features.

![[Pasted image 20231121103444.png]]

Which of these lines would be the best separator for our case ? Obviously, it is A, since it delimits the best the 2 groups. But what about this case ?

![[Pasted image 20231121103654.png]]

Well, it would still be A. Since B and C are so close to some already existing points, it would make the accuracy worst.

![[Pasted image 20231121104605.png]]

The dotted line in our graph represent the **margin** of our vector, which corresponds to the closest point to the vector. Our goal is to have the largest margins possible.

![[Pasted image 20231121104855.png]]

Our points used to calculate the margin are called support vectors.
The issue with SVM is that they are very bad against outliers (for example a lone point on the other side)

![[Pasted image 20231121105659.png]]

In the case of a one dimensional dataset, we can find ways to make it 2D, and thus finding more easily a vector that is more appropriate.
The transformation that we just did here is known as a **kernel trick**.
$$ x \to (x_1 x^2) $$
#### Implementation

```python
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
print()
print("SVM")
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support  

#            0       0.81      0.79      0.80      1369
#            1       0.88      0.90      0.89      2435

#     accuracy                           0.86      3804
#    macro avg       0.85      0.84      0.85      3804
# weighted avg       0.86      0.86      0.86      3804
```
