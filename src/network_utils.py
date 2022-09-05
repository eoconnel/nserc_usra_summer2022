import numpy as np
import matplotlib.pyplot as plt
import PIL
import pickle
from . import network

def generate_image_data(image, n, size):
    im = PIL.Image.open(image).convert('L').resize(size)
    im = np.array(im)
    im = np.rint(im / 255)
    
    data = np.random.randint(0, [size[1], size[0]], size=[n, 2, 1])
    labels = np.array([int(im[x[0], x[1]]) for x in data]).reshape(n,1,1)
    data = np.array([ np.multiply(x, [[1/size[1]], [1/size[0]]]) for x in data])

    return data, labels

def generate_hyperplanes(net):
    weights = net.layers[0].weights
    biases = net.layers[0].biases
    
    nets = []
    for i in range(len(biases)):
        nets.append(network.Network([network.Linear(2,1), network.Sigmoid()]))
        nets[i].layers[0].weights = weights[i].reshape(1,2)
        nets[i].layers[0].biases = biases[i].reshape(1,1)
    
    return nets

def visualize_2d_function(func, n, convert=True):
    ii = np.arange(0, n[0])
    jj = np.arange(0, n[1])
    output = np.zeros([n[0], n[1]])
    for i in ii:
        for j in jj:
            if convert:
                output[i, j] = func(np.reshape([[j/n[1]], [1-i/n[0]]], (2, 1)))
            else:
                output[i, j] = func(np.reshape([[i/n[0]], [j/n[1]]], (2,1)))
    return output

def visualize_3d_function(func, n, convert=True):
    ii = np.arange(0, n)
    jj = np.arange(0, n)
    kk = np.arange(0, n)
    output = np.zeros([n, n, n])
    for i in ii:
        for j in jj:
            for k in kk:
                if convert:
                    output[i, j, k] = func(np.reshape([[j/n], [1-i/n], [k/n]], (3, 1)))
                else:
                    output[i, j, j] = func(np.reshape([[i/n], [j/n], [k/n]], (3,1)))
    return output

def read_network(filename):
    with open(filename, 'rb') as f:
        network_dict = pickle.load(f)
        
    layers = []
    for layer in network_dict['layers']:
        layer_type = layer['type']
    
        if layer_type == 'Linear':
            weights = layer['weights']
            biases = layer['biases']
            net_layer = network.Linear(weights.shape[1], weights.shape[0])
            net_layer.weights = weights
            net_layer.biases = biases
        elif layer_type == 'Convolutional':
            kernels = layer['kernels']
            biases = layer['biases']
            input_shape = layer['input_shape']
            num_kernels = kernels.shape[0]
            kernel_size = kernels.shape[2]
            net_layer = network.Convolutional(input_shape, kernel_size, num_kernels)
            net_layer.kernels = kernels
            net_layer.biases = biases
        elif layer_type == 'Reshape':
            net_layer = network.Reshape(layer['input_shape'], layer['output_shape'])
        elif layer_type == 'Rescale':
            net_layer = network.Rescale(layer['factor'])
        elif layer_type == 'Sigmoid':
            net_layer = network.Sigmoid()
        elif layer_type == 'ReLU':
            net_layer = network.ReLU()
        elif layer_type == 'Step':
            net_layer = network.Step()
        elif layer_type == 'Summation':
            net_layer = network.Summation()
        
        layers.append(net_layer)
        
    return network.Network(layers)

def save_network(net, filename):
    network_dict = {'layers': []}
    for layer in net.layers:
        layer_type = type(layer).__name__
        dict_layer = {'type': layer_type}
        
        if layer_type == 'Linear':
            dict_layer['weights'] = layer.weights
            dict_layer['biases'] = layer.biases
        elif layer_type == 'Convolutional':
            dict_layer['kernels'] = layer.kernels
            dict_layer['biases'] = layer.biases
            dict_layer['input_shape'] = layer.input_shape
        elif layer_type == 'Reshape':
            dict_layer['input_shape'] = layer.input_shape
            dict_layer['output_shape'] = layer.output_shape
        elif layer_type == 'Rescale':
            dict_layer['factor'] = layer.factor
            
        network_dict['layers'].append(dict_layer)
    
    with open(filename, 'wb') as f:
        pickle.dump(network_dict, f)
