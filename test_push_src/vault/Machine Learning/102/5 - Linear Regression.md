In a **Linear Regression**, we want to take a set of data and fit a **Linear Model** in it.

![[Pasted image 20231121150319.png]]

The line, known as the **Line of Best Fit**, can be represented like this:
$$ y = b_0 + b_1 x $$
$$ b_0: \text{The point where the line meets x=0} $$
$$ b_1 : \text{Defines the slope of the line} $$
- **Residual** (sometimes called **error**): How far off is our prediction from a data point that we already have

![[Pasted image 20231121151101.png]]
$$ residual = |y_i - \hat{y_i}| $$
Our goal is to find the smallest residual value. The way to do this is to minimize the sum of the residuals using the lowest value of b0 and b1.
This is known as **simple linear regression**.
$$ y = b_0 + b_1 x $$
If we have more than one value in our feature vector, then this become a **multiple linear regression**.
$$ y = b_0 + b_1 x_1 + b_2 x_2 + ... + b_{n} x_n $$
## Assumptions

- **Linearity**: Does my data follow a linear pattern ? Does x increases as the time as y increases ? Linearity wouldn't be satisfied if the line of best fit is curved.
- **Independence**: A point in the dataset shouldn't influence on another point.
- **Normality**: The residual plot should follow a normal distribution
- **Homoskedasticity**: If the spread on the residual plot is not constant, this assumption would not be fulfilled. (This would mean that using Linear Regression would not be appropriate)

![[Pasted image 20231122113204.png]]

## How to evaluate a Linear Regression Model 

### Mean Absolute Error (MAE)
$$ \text{MAE} = \frac{\sum_\limits{i = 1}^{n} | y_i - \hat{y_i}|}{n} $$
![[Pasted image 20231122113822.png]]

This gives us the average between our predicted value and ou training dataset. We can then say `This is by how much our data is off`.

### Mean Squared Error (MSE)
$$ \text{MSE} = \frac{\sum_\limits{i = 1}^{n} ( y_i - \hat{y_i})^2}{n} $$
This would give us by how much *squared* our data is off, which is not very human understandable. Which is why we created...

### Root Mean Squared Error (RMSE)
$$ \text{RMSE} = \sqrt{\frac{\sum_\limits{i = 1}^{n} ( y_i - \hat{y_i})^2}{n}} $$
Now, we can have a much more readable result for a human.

### Coefficient of Determination (R^2)
$$ R^2 = 1 - \frac{RSS}{TSS} $$
- **RSS**: Squared of the sum residual (What is our error with our respect to the line of best fit)
$$ RSS = \sum\limits_{i = 1}^{n} (y_i - \hat{y_i})^2 $$
- **TSS**: Total sum of squared (What is our error compared to the average y value)
$$ TSS = \sum\limits_{i = 1}^{n} (y_i - \bar{y})^2 $$
*the y-hat (called mean) is equal to the average value of each points. We want to calculate how far off are all point from that line.*

![[Pasted image 20231122115252.png]]

When RSS is smaller than TSS, R has the tendencies to go towards 1, which means that we have a good predictor.

#### R^2 adjusted

What this does is that it adjusts R to the number of terms that we have. The value of R^2 adjusted increases if the new term improve the model fit more than expected.