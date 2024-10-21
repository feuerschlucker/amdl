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
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Use tf.data.Dataset to prepare data
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=32)

######################################################################
# Alternative Approach 1
######################################################################

inputs = tf.keras.Input(shape=(28, 28))
flattened_inputs = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(flattened_inputs)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
net = tf.keras.Model(inputs=inputs, outputs=outputs)
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["categorical_accuracy"])
net.fit(train_ds, epochs=3)
print('Testing...')
net.evaluate(test_ds)


######################################################################
# Alternative Approach 2
######################################################################

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.d2 = tf.keras.layers.Dense(units=10)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


net = NeuralNetwork()
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["categorical_accuracy"])
net.fit(train_ds, epochs=3)
print('Testing...')
net.evaluate(test_ds)


######################################################################
# Alternative Approach 3
######################################################################

net = NeuralNetwork()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
num_epochs = 3

# Model training
for epoch in range(num_epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    train_accuracy.reset_state()

    # Training loop
    for image_batch, label_batch in train_ds:
        with tf.GradientTape() as tape:
            predictions = net(image_batch, training=True)   # Forward Pass
            loss = loss_object(label_batch, predictions)

        gradients = tape.gradient(loss, net.trainable_variables)  # Backward Pass
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        train_loss(loss)
        train_accuracy(label_batch, predictions)

    print(f'Epoch {epoch + 1}, '
          f'Training Loss: {train_loss.result()}, '
          f'Training Accuracy: {train_accuracy.result() * 100}, ')

# Testing
for test_image_batch, test_label_batch in test_ds:
    predictions = net(test_image_batch, training=False)
    loss = loss_object(test_label_batch, predictions)
    test_loss(loss)
    test_accuracy(test_label_batch, predictions)

print('Testing...')
print(f'Testing Loss: {test_loss.result()}, Testing Accuracy: {test_accuracy.result() * 100}')
