import pickle
from neuralnetworksanddeeplearning.src import mnist_loader, network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 100, 10])
net.SGD(training_data, 20, 20, 3, test_data=test_data)
with open('pesos.pickle', 'wb') as f:
    pickle.dump(net, f)

