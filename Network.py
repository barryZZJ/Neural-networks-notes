#Network类代表一个神经网络
import random
import numpy as np
import mnist_loader
class Network(object):

    def __init__(self, sizes):
        '''
        :param sizes: The number of neurons in the respective layers. e.g.:a Network object with 2-3-1 neurons in 1st, 2nd, 3rd layer, it should be [2,3,1].
        '''
        self.num_layers=len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def calculate(self, a):
        '''returns activation value of the final layer.
        "a" is an (n,1) ndarray
        '''
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self,training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data=list(training_data)
        test_data=list(test_data)
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        '''``x`` in mini_batch is vertical vector by default'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            single_nabla_b, single_nabla_w = self.backprop(x,y)
            nabla_b = [nb+snb for nb, snb in zip(nabla_b, single_nabla_b)]
            nabla_w = [nw+snw for nw, snw in zip(nabla_w, single_nabla_w)]

        self.biases = [b-eta/len(mini_batch)*nb for b, nb in zip(self.biases,nabla_b)]
        self.weights = [w-eta/len(mini_batch)*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        tmp = self.cost_deriv(activation,y) * sigmoid_prime(z)
        nabla_b[-1]=tmp
        nabla_w[-1]=np.dot(tmp, activations[-2].transpose())

        for t in range(2,self.num_layers):
            tmp = np.dot(self.weights[-t+1].transpose(), tmp) * sigmoid_prime(zs[-t])
            nabla_b[-t] = tmp
            nabla_w[-t] = np.dot(tmp, activations[-t-1].transpose())
        return (nabla_b,nabla_w)

    def cost_deriv(self,FLactivation,y):
        return (FLactivation - y)

    def evaluate(self,test_data):
        test_results=[(np.argmax(self.calculate(x)),y)
                      for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

def sigmoid(z):
    '''
    :param z: accept list as input, output list as well.
    :return:
    '''
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    tmp = sigmoid(z)
    return tmp*(1-tmp)

train,vali,test=mnist_loader.load_data_wrapper()
net = Network([784,30,10])
#print(net.evaluate(test),"/",len(test))
net.SGD(train,30,10,3.0,test)