from utils import mnist_reader

x_train, y_train = mnist_reader.load_mnist("data/fashion", kind="train")
x_test, y_test = mnist_reader.load_mnist("data/fashion", kind="t10k")
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print(type(x_train), type(x_test), type(y_train), type(y_test))
