![[Pasted image 20231121110917.png]]

In a neural network, each input corresponds to a feature, which then get weighted using a **w** value. Then, the sum of all of these get computed in a **neuron**. To add some randomness in the neural network, we will add a **bias** to the to the neuron, which will shift some values.
This will then go into what's called an **activation function**, which will then give us our output.

![[Pasted image 20231121111249.png]]

## The activation function

If we don't have an activation function, running a neural network would just change values in a linear manner, which we could just do using a formula or a simpler algorithm.

**Without an activation function, this would just be a linear model !**

Here are some examples of an activation function:

![[Pasted image 20231121111721.png]]

If we take a look at our L2 loss function, here is what it would look like:

![[Pasted image 20231121112013.png]]

Once we get far away from our center, the error becomes highly unforgivable. Thus, our goal becomes to go down. We can use something called **gradient descent**.

![[Pasted image 20231121112450.png]]

Depending on where we are on the curve, the value of the gradient descent vector changes.
To calculate how much we have to backstep, we need to look up **back propagation**.
$$ w_{0, new} = w_{0, old} + \alpha * {arrowValue} $$
This way, we will change the weight of the feature to make it closer to what we want.
The alpha letter (called the **learning rate**) is here to minimize the error while we step down, letting us make small step by small step, in case we're wrong.

*The reason it is a + in this formula is because we want the negative gradient*

All of that is what corresponds to the adjustments made to the model during the training.

## Implementation

```python
import tensorflow as tf

# ------------------ Functions for visualisation ------------------
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
# -----------------------------------------------------------------

# Layers of the neural network 
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])  

history = nn_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

plot_loss(history)
plot_accuracy(history)
```

![[Pasted image 20231121120301.png]]
![[Pasted image 20231121120322.png]]

## Optimize

Let's modify our code a bit to loop through an array of parameters, and find the best performing model.

```python
import tensorflow as tf

# ------------------ Functions for visualisation ------------------
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()
# -----------------------------------------------------------------

def train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),           # prevent overfitting
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),           # prevent overfitting
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    return nn_model, history


least_val_loss = float('inf')
least_loss_model = None
epochs = 100
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, .2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch_size {batch_size}")
                model, history = train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history)
                val_loss = model.evaluate(x_valid, y_valid, verbose=0)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model
```

```
Best model
Model: "sequential_52"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense_156 (Dense)           (None, 64)                704

 dropout_104 (Dropout)       (None, 64)                0

 dense_157 (Dense)           (None, 64)                4160

 dropout_105 (Dropout)       (None, 64)                0

 dense_158 (Dense)           (None, 1)                 65

=================================================================
Total params: 4929 (19.25 KB)
Trainable params: 4929 (19.25 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
