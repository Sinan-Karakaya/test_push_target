Now, instead of predicting a category, we're actually going to predict a value.
We'll use this dataset:

[Seoul Bike Sharing Demand - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

## Implementation 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression

ds_cols = ['bike_count', 'hour', 'temp', 'humidity', 'wind', 'visibility', 'dew_pt_temp', 'radiation', 'rain', 'snow', 'functional']
df = pd.read_csv('./sample_data/SeoulBikeData.csv').drop(['Date', 'Holiday', 'Seasons'], axis=1)
df.columns = ds_cols
df['functional'] = (df['functional'] == 'Yes').astype(int)

# Keep only after 12:00
df = df[df['hour'] == 12]
df = df.drop(['hour'], axis=1)

# Show how the bike_count change depending on the onther features
if False:
    for label in df.columns[1:]:
        plt.scatter(df[label], df['bike_count'])
        plt.title(label)
        plt.ylabel('Bike Count at noon')
        plt.xlabel(label)
        plt.show()

# We get rid of wind, visibility and functional since they would not be useful for prediction
df = df.drop(['wind', 'visibility', 'functional'], axis=1)

# Train/validation/test dataset
train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

def get_xy(df, y_label, x_label=None):
    dataframe = copy.deepcopy(df)
    if x_label is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values 
    else:
        if len(x_label) == 1:
            X = dataframe[x_label[0]].values.reshape(-1, 1)
        else:
            X = dataframe[x_label].values
  
    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))
    return data, X, y

_, X_train_temp, y_train_temp = get_xy(train, 'bike_count', x_label=['temp'])
_, X_val_temp, y_val_temp = get_xy(val, 'bike_count', x_label=['temp'])
_, X_test_temp, y_test_temp = get_xy(test, 'bike_count', x_label=['temp'])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)

# This gives us the R squared
# From here, we can get the line of best fit using the predict function
print(temp_reg.score(X_train_temp, y_train_temp)) 

# ------------------- Mulitple Linear Regression -------------------
# We keep all the labels except bike_count
_, X_train_all, y_train_all = get_xy(train, 'bike_count', x_label=df.columns[1:])
_, X_val_all, y_val_all = get_xy(val, 'bike_count', x_label=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, 'bike_count', x_label=df.columns[1:])

reg_all = LinearRegression()
reg_all.fit(X_train_all, y_train_all)
reg_all.score(X_test_all, y_test_all)

# ------------------- Regression with Neural Network -------------------
temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(units=1)
])

temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

history = temp_nn_model.fit(X_train_temp.reshape(-1), y_train_temp, epochs=1000, verbose=0, validation_data=(X_val_temp, y_val_temp))

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()

# plot_loss(history)

# ------------------- Multiple nodes -------------------
temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history = nn_model.fit(X_train_temp, y_train_temp, epochs=100, verbose=0, validation_data=(X_val_temp, y_val_temp))

# plot_loss(history)

# Fig.1
# plt.scatter(X_train_temp, y_train_temp, label='Data', color='blue')
# x = tf.linspace(-20, 40, 100)
# plt.plot(x, nn_model.predict(np.array(x).reshape(-1, 1)), label='Fit', color='red', linewidth=3)
# plt.legend()
# plt.title("Bike vs temperature")
# plt.ylabel("Bike count at noon")
# plt.xlabel("Temperature")
# plt.show()

# ------------------- Multiple features -------------------
all_normalizer = tf.keras.layers.Normalization(input_shape=(6,), axis=-1)
all_normalizer.adapt(X_train_all)

nn_model = tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history = nn_model.fit(X_train_all, y_train_all, epochs=100, verbose=0, validation_data=(X_val_all, y_val_all))

# Fig.2
plot_loss(history)

def get_mse(y_pred, y_real):
    return (np.square(y_pred - y_real)).mean()

y_pred_lr = reg_all.predict(X_test_all)
y_pred_nn = nn_model.predict(X_test_all)

print(get_mse(y_pred_lr, y_test_all)) # ~89000
print(get_mse(y_pred_nn, y_test_all)) # ~85000

# Fig.3
ax = plt.axes(aspect='equal')
plt.scatter(y_test_all, y_pred_lr, label='Lin Reg Preds')
plt.scatter(y_test_all, y_pred_nn, label='NN Preds')
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 1800]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
_ = plt.plot(lims, lims, c='red')
plt.show()
```

## Results

*Neural model trained on only one input, the temperature*
![[Pasted image 20231122150704.png]]

With a neural model, we don't get a linear result anymore.

#### Multiple inputs

In our results above, we only used the temperature as a feature for our model. Let's try now with multiple features.

![[Pasted image 20231122153209.png]]

We get a non linear MSE.
Now let's compare both our Linear Regression model and our Neural Network model.

![[Pasted image 20231122153253.png]]

The NN model seems to be more accurate than our LR model.