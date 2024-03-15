import numpy as np
from statistics import mode


def random_compress(x_train, x_test, new_dimension):
    k = x_train.shape[1]
    w = np.random.normal(size=(k, new_dimension))

    x_train = x_train @ w
    x_test = x_test @ w
    return x_train, x_test


# def single_predict(x_train, y_train, test_image, n=5):
#     distances = []
#     for i in range(len(x_train)):
#         d = np.linalg.norm(x_train[i] - test_image, ord=2.0)
#         distances.append(d)
#     min_distances_indexes = np.argsort(distances)
#     distances.sort()
#     labels = y_train[min_distances_indexes[:n]]
#     prediction = mode(labels)
#     return prediction
