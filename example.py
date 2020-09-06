import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Model import Model
from Layers import Dense, ReLU


def flatten(array):
    flat = []
    for i in range(len(array)):
        flat.append(array[i].flatten())
    return np.array(flat)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_flat, x_test_flat = flatten(x_train), flatten(x_test)

for i in np.random.randint(len(x_train), size=8):
    plt.title("Label: %i" % y_train[i])
    plt.imshow(x_train[i], cmap='Blues')
    plt.show()

model = Model()
model.add(Dense(100), ReLU(), Dense(200), ReLU(), Dense(10))
model.train(x_train_flat, y_train, 32, 10, x_test_flat, y_test)
