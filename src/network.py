import numpy as np
import random
from scipy import signal

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, x):
        pass
    
    def backward(self, nabla_out):
        pass
    
    def update_params(self, eta, n):
        pass

class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.biases = np.random.randn(output_size, 1)
        self.nabla_w = 0
        self.nabla_b = 0
        
    def forward(self, x):
        self.input = x
        return np.dot(self.weights, self.input) + self.biases
    
    def backward(self, nabla_out):
        self.nabla_w += np.dot(nabla_out, self.input.T)
        self.nabla_b += nabla_out
        return np.dot(self.weights.T, nabla_out)
    
    def update_params(self, eta, n):
        self.weights -= eta * self.nabla_w / n
        self.biases -= eta * self.nabla_b / n
        self.nabla_w = 0
        self.nabla_b = 0

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, num_kernels):
        self.num_inputs = input_shape[0]
        self.num_kernels = num_kernels
        self.input_shape = input_shape
        self.output_shape = (num_kernels, input_shape[1] - kernel_size + 1, input_shape[2] - kernel_size + 1)
        self.kernel_shape = (num_kernels, input_shape[0], kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)
        self.nabla_k = 0
        self.nabla_b = 0
        
    def forward(self, x):
        self.input = x
        y = np.zeros(self.output_shape)
        for i in range(self.num_kernels):
            for j in range(self.num_inputs):
                y[i] += signal.correlate2d(self.input[j], self.kernels[i][j], mode='valid')
        return y + self.biases 
    
    def backward(self, nabla_out):
        nabla_k = np.zeros(self.kernel_shape)
        nabla_x = np.zeros(self.input_shape)
        for i in range(self.kernel_shape[0]):
            for j in range(self.input_shape[0]):
                nabla_k[i][j] = signal.correlate2d(self.input[j], nabla_out[i], mode='valid')
                nabla_x[j] += signal.convolve2d(nabla_out[i], self.kernels[i][j], mode='full')
        
        self.nabla_k += nabla_k
        self.nabla_b += nabla_out
        return nabla_x

    def update_params(self, eta, n):
        self.kernels -= eta * self.nabla_k / n
        self.biases -= eta * self.nabla_b / n
        self.nabla_k = 0
        self.nabla_b = 0
    
class Sigmoid(Layer):
    def __init__(self):
        self.func = lambda x: 1 / (1 + np.exp(-x))
        self.func_prime = lambda x: self.func(x) * (1 - self.func(x))
        
    def forward(self, x):
        self.input = x
        return self.func(self.input)
    
    def backward(self, nabla_out):
        return np.multiply(nabla_out, self.func_prime(self.input))
    
    def update_params(self, eta, n):
        pass

class ReLU(Layer):
    def __init__(self):
        self.func = lambda x: x * (x > 0)
        self.func_prime = lambda x: 1.0 * (x > 0)
        
    def forward(self, x):
        self.input = x
        return self.func(self.input)
    
    def backward(self, nabla_out):
        return np.multiply(nabla_out, self.func_prime(self.input))
    
    def update_params(self, eta, n):
        pass

class Step(Layer):
    def __init__(self):
        self.func = lambda x: np.heaviside(x, 1)
        self.func_prime = lambda x: np.zeros(x.shape)
        
    def forward(self, x):
        self.input = x
        return self.func(self.input)
    
    def backward(self, nabla_out):
        return self.func_prime(self.input)
    
    def update_params(self, eta, n):
        pass

class Summation(Layer):
    def __init__(self):
        pass
        
    def forward(self, x):
        self.input = x
        return np.sum(self.input, axis=(1,2)).reshape(self.input.shape[0], 1)
    
    def backward(self, nabla_out):
        in_shape = self.input.shape
        nabla_in = np.ones(in_shape)
        return np.array([nabla_in[i] * nabla_out[i] for i in range(len(nabla_out))])
    
    def update_params(self, eta, n):
        pass

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, x):
        return np.reshape(x, self.output_shape)
    
    def backward(self, nabla_out):
        return np.reshape(nabla_out, self.input_shape)

    def update_params(self, eta, n):
        pass

class Rescale(Layer):
    def __init__(self, factor):
        self.factor = factor
        
    def forward(self, x):
        self.input = x
        return x[:, ::self.factor, ::self.factor]
    
    def backward(self, nabla_out):
        return nabla_out.repeat(self.factor, axis=1).repeat(self.factor, axis=2)
    
    def update_params(self, eta, n):
        pass

class Network:
    def __init__(self, layers):
        self.layers = layers
        
    def forward_prop(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y
    
    def backward_prop(self, nabla):
        for layer in reversed(self.layers):
            nabla = layer.backward(nabla)
        return nabla
            
    def update_params(self, eta, n):
        for layer in self.layers:
            layer.update_params(eta, n)
    
    def train(self, input_data, labels, mini_batch_size, eta, epochs, epoch_disp, evaluation='error',
        test_data=None, test_labels=None):
        n = len(input_data)
        
        for e in range(epochs):

            p = np.random.permutation(n)
            input_data = input_data[p]
            labels = labels[p]
            
            mini_batch_inputs = [input_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batch_labels = [labels[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for i in range(len(mini_batch_inputs)):
                
                mini_batch_input = mini_batch_inputs[i]
                mini_batch_label = mini_batch_labels[i]
                
                for x, y in zip(mini_batch_input, mini_batch_label):
                    output = self.forward_prop(x)
                    nabla = self.mse_prime(output, y)
                    self.backward_prop(nabla)
            
                self.update_params(eta, mini_batch_size)
            
            if (test_data is not None) & (epoch_disp != 0):
                
                if (e % epoch_disp == 0) | (e == epochs - 1):

                    if evaluation == 'error':
                        err = self.evaluate_error(test_data, test_labels)
                        print(f'Error in epoch {e+1} / {epochs}: {round(err, 5)}')
                    elif evaluation == 'percentage':
                        acc = self.evaluate_percentage(test_data, test_labels)
                        print(f'Accuracy in epoch {e+1} / {epochs}: {round(acc, 2)}%')
                    elif evaluation == 'percentage_onehot':
                        acc = self.evaluate_percentage_onehot(test_data, test_labels)
                        print(f'Accuracy in epoch {e+1} / {epochs}: {round(acc, 2)}%')

    def evaluate_percentage(self, test_data, labels):
        correct = 0
        for x, y in zip(test_data, labels):
            output = int(np.rint(self.forward_prop(x)))
            correct += 1 * (output == int(y))
        return 100*(correct / len(labels))

    def evaluate_percentage_onehot(self, test_data, labels):
        correct = 0
        for x, y in zip(test_data, labels):
            output = int(np.argmax(self.forward_prop(x)))
            correct += int(y[output])
        return 100*(correct / len(labels))

    def evaluate_error(self, test_data, labels):
        err = 0
        for x, y in zip(test_data, labels):
            err += self.mse(self.forward_prop(x), y)
        return err/len(labels)
    
    def mse(self, output, actual):
        return np.mean(np.power(output - actual, 2))
    
    def mse_prime(self, output, actual):
        return (2/np.size(output)) * (output - actual)