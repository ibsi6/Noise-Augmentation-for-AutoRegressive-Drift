import numpy as np
from tensorflow.keras.datasets import mnist

def load_dataset():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = np.concatenate([x_train, x_test], axis=0)
    x_train = x_train.astype(np.float32) / 255.0  # Normalize to [0, 1]
    x_train = x_train * 2 - 1  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, -1)  # Add channel dimension
    return x_train

def data_loader(dataset, batch_size):
    num_batches = len(dataset) // batch_size
    perm = np.random.permutation(len(dataset))
    for i in range(num_batches):
        batch = dataset[perm[i * batch_size: (i + 1) * batch_size]]
        yield {'x': batch}