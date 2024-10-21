#######################################################
# Advanced Machine & Deep Learning (190.017) WS24
# An introductory tutorial on MNIST classification
# Author: ozan.oezdenizci@unileoben.ac.at
#######################################################
import tensorflow as tf
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)


# Load and prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Plot some training samples
plot_some_x = x_train[:10]
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(plot_some_x[i], cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Configure the model layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

# Making predictions (forward pass) with individual data samples
predictions = model(x_train[:1]).numpy()

# Configure the model training procedure
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Perform model training
model.fit(x_train, y_train, epochs=20, batch_size=64)

# Evaluate trained model on the test set
model.evaluate(x_test, y_test, batch_size=64)
