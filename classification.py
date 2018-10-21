# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('Working with TensorFlow version %s' % tf.__version__)

# Read fashion_mnist data set
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images
plt.figure(figsize=(10, 10))
for index in range(25):
    plt.subplot(5, 5, index + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[index], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[index]])
plt.show()

# Building models using Keras
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compiling models
model1.compile(optimizer=tf.train.AdamOptimizer(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model2.compile(optimizer=tf.train.AdamOptimizer(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Training our models
model1.fit(train_images, train_labels, epochs=5)

model2.fit(train_images, train_labels, epochs=5)

# Evaluating accuracy
test_loss, test_acc = model1.evaluate(test_images, test_labels)
print('Test accuracy for Model 1:', test_acc)

test_loss, test_acc = model2.evaluate(test_images, test_labels)
print('Test accuracy for Model 2:', test_acc)





