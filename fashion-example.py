import numpy as np
import fashionKNN as fast
from utils import mnist_reader

x_train, y_train = mnist_reader.load_mnist("data/fashion", kind="train")
x_test, y_test = mnist_reader.load_mnist("data/fashion", kind="t10k")

x_train, x_test = fast.random_compress(x_train, x_test, 64)


y_train_vectors = np.zeros((len(y_train), 10), dtype=float)
for i in range(len(y_train_vectors)):
    count = y_train[i]
    y_train_vectors[i][count] = 1


def single_predict(x_train, y_train, test_image, n=5):
    vector_sum = np.zeros(10, dtype=float)
    distances = []
    for i in range(len(x_train)):
        d = np.linalg.norm(x_train[i] - test_image, ord=2.0)
        distances.append(d)
    min_distances_indexes = np.argsort(distances)
    distances.sort()
    # print(distances[:n])
    labels = y_train_vectors[min_distances_indexes[:n]]
    # print(len(labels))
    for p in range(len(labels)):
        labels[p] = labels[p] / distances[p]
    for j in range(len(labels)):
        vector_sum += labels[j]
        # vector_sum = vector_sum
    # print(vector_sum)
    vector_sum = np.argsort(vector_sum)
    # print(vector_sum)
    prediction = vector_sum[-1]
    # print(y_train[prediction])
    return prediction
    # return labels


prediction = np.zeros((len(y_test),))
correct = 0
for i in range(len(y_test)):
    if i % 10 == 0:
        print(i, correct)
    prediction[i] = single_predict(x_train, y_train, x_test[i])
    if prediction[i] == y_test[i]:
        correct += 1


prediction_accuracy = (correct / len(y_test)) * 100
print(prediction_accuracy)
# prediction = np.zeros((len(y_test),))
# correct = 0
# for i in range(len(y_test)):
#     if i % 10 == 0:
#         print(i, correct)
#     prediction[i] = fast.single_predict(x_train, y_train, x_test[i])
#     if prediction[i] == y_test[i]:
#         correct += 1

# prediction_accuracy = (correct / len(y_test)) * 100
# print(prediction_accuracy)
