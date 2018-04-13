import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
print (X_train.shape[0], X_train.shape[1])
print (y_train.shape[0], y_train.shape[1])