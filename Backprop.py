# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:17:06 2019

@author: asadc
"""
'''
Backprop class takes a network model and data point to calculate
the gradient descent based on the error in each layer. It starts calculating
the error for the output layer first and then goes backwords for each layer.
'''

import numpy as np
import Metrics as mt

class Backprop:           
                
        def backprop(self, x, y, biases, weights, num_layers):
                #takes network weights and biases list as input and a point to calculate error.
                metrics = mt.Metrics()
                activation = x          #current activation layer is the input
                activations = [x]       #list to hold activations of layers
                z_vec = []              #list to hold the z_vec = sum(weights*x_i)
                #initialize weights and biases to 0 to hold the delta error.
                delta_weights = []
                delta_biases = []
                for i in biases:        delta_biases.append(np.zeros(i.shape))
                for j in weights:       delta_weights.append(np.zeros(j.shape))
                
                #run the current network to get activations and z_vec for each layer and appent to the list.
                for b, w in zip(biases, weights):
                        z = np.dot(w, activation)+ b
                        z_vec.append(z)
                        activation = metrics.sigmoid(z)
                        activations.append(activation)
                #start for the last layer(output layer) to calculate the cost derivative and sigmoid derivative
                #the derivative functions are defined in Metrics class and Backprop is using an instance of
                #Metrices class.
                delta_last_layer = metrics.cost_derivative(activations[-1], y) * metrics.sigmoid_derivative(z_vec[-1])
                #delta bias is same as the delta error for the same layer.
                delta_biases[-1] = delta_last_layer
                #calculate delta weights for the last layer
                delta_weights[-1] = np.dot(delta_last_layer, activations[-2].transpose())
                
                #we have the last layer error gradient. Now going backwords with each layer and 
                #use the already calculated delta error layer to calculate the existing layer delta weights
                #and biases. The following loop goes backword starting from 2nd last layer. (already calculated)
                #the gradient for the final output layer.
                prev_delta_error = delta_last_layer
                for l in range(2, num_layers):
                        z = z_vec[-l]
                        sig_derivative = metrics.sigmoid_derivative(z)
                        delta_error = np.dot(weights[-l+1].transpose(), prev_delta_error) * sig_derivative
                        delta_biases[-l] = delta_error
                        delta_weights[-l] = np.dot(delta_error, activations[-l-1].transpose())
                        prev_delta_error = delta_error
                        
                return (delta_biases, delta_weights) #return calculated delta weights and biases.
