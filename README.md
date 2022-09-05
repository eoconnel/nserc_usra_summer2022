# nserc_usra_summer2022
Neural network code and research conducted as an NSERC Undergraduate Student Research Award recipient.

## Author:  
Ethan O'Connell  
eoconnel@unb.ca  
University of New Brunswick  
Bachelor of Science with Honours in Mathematics, 5th year  

## Supervisor:  
Nicholas Touikan  
nicholas.touikan@unb.ca  
University of New Brunswick, faculty of Mathematics and Statistics  

## Description of Research
This repository contains a write up of my code and research conducted from May - August 2022 as a research assistant and recipient of the NSERC Undergraduate Student Research Award at UNB.  

The main objective of the summer research project was to conduct an exploratory analysis of neural networks, with a particular focus on shallow neural networks and convolutional networks and the math involved in constructing these networks. 
A future goal of this project would be to construct a fully analog neural network which would act as a functional on an arbitrary function space. 
For example, a network could be constructed which takes as input a function over the group of rotations in two-dimensional Euclidean space, SO(2), and returns some scalar real value.  

To facilitate the exploratory analysis of neural networks, a fully modular network was written in python along with several methods to create and test various network architectures. A series of experiments were also conducted as explained below.

## Contents
Most of the experimentation was conducted using jupyter notebooks to facilitate visualization and testing. The contents of each of the notebooks and python modules are explained below:  
* __network.py__: neural network class as explained in `modular_neural_network.ipynb`.  
* __network_utils.py__: utility functions to facilitate neural network experimentation.  
* __modular_neural_network.ipynb__: explanation of the `network.py` module.  
* __hyperparameter_tuning.ipynb__: a set of experiments conducted to determine an approximate learning rate and number of epochs for the binary classification test problems.  
* __binary_classification.ipynb__: an explanation of classification problems and training a series of networks over binary classification regions generated using hand-drawn black and white images.  
* __concave_classification.ipynb__: an extension of binary classification problems to more complex regions involving concavity.  
* __function_approximation.ipynb__: an introduction to function approximation problems and a demonstration of one and two dimensional cases.  
* __convolutional_networks.ipynb__: an explanation of convolutional networks implemented over the MNIST hand-written digit classification problem as well as an attempted fully convolutional network.  
* __universal_approximation.ipynb__: a constructive proof and visual demonstration of the Universal Approximation Theorem for neural networks.  

## Resources
Below are some of the resources consulted throughout the research project.  
* Neural Network from Scratch | Mathematics & Python Code: https://youtu.be/pauPCy_s0Ok  
* Convolutional Neural Network from Scratch | Mathematics & Python Code: https://youtu.be/Lakz2MoHy6o  
* Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/index.html  
* Deep Learning: https://www.deeplearningbook.org
* Approximation by Superpositions of a Sigmoidal Function: https://link.springer.com/content/pdf/10.1007/BF02551274.pdf  
* Volume of a hypersphere: https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/HypersphereVolume.pdf  
* Area and Volume of a Hyperspherical Cap: https://scialert.net/fulltext/?doi=ajms.2011.66.70  
* Evenly distributing n points on a sphere: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere   
