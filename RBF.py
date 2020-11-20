# -*- coding: utf-8 -*-
"""
@author: cnoble

During initialization, layers are intialized, bias and weights are randomly intialized,
    and learning rate is set.
In train(), feedforward() is called for each value in the training set, and backprop() 
    is called from the Backprop class to train the instance and get weights adjustments. 
    Weights and bias are updated at the end of each epoch. This repeats until convergence
    or until the max iterations are reached.
In feedfoward(), gaussianActivation() is called for the combination of the input node 
    with each hidden node. The activated node combinations are returned.
In gaussianActivation(), the euclideanDistance() from the Metrics class is used to find 
    the distance between the given nodes. The activation of the two given nodes is 
    calculated with the Gaussian Activation Function.
In convertOutput(), the expected output is converted to an array containing the 
    single regression value for regression, or a vector or 1's and 0's for classification.
In weightUpdate(), the weights and bias are updated by averaging the results from
    each iteration of backprop, and multiplied by the learning rate.
In test(), feedforward(), calcOutput() and predict() are called to return predictions
    for each point in the test set.
In calcOutput() the output layer is calculated by taking the dot product of the weights 
    with the activation results. Predictions are returned by calling predict().
In predict() predictions are returned. For regression, the output node will 
    contain the predicted regression value. For classification, the predicted class
    is the class represented by the output node containing the highest value.     

"""
from sklearn.model_selection import train_test_split
import numpy as np
from math import exp
import Metrics as mt
import Backprop

class RBF:
    # input parameters: training set, prototype set, T/F classification/regression, 
    # number of  many classes for classification (set to 1 for regression)
    def __init__(self, trainSet, prototypeSet, isClassification, numberOfClasses):
        self.classification = isClassification
        self.numOutNodes = numberOfClasses
        self.metrics = mt.Metrics()
        self.backpropLayers = 2
        
        # initialize layers
        self.inputLayer, self.valSet = train_test_split(trainSet, test_size = 0.2, random_state = 0)
        self.outputLayer = []
        
        # drop class column
        self.inputClass = np.array(self.inputLayer[self.inputLayer.columns[-1]])
        self.inputLayer = self.inputLayer.drop([self.inputLayer.columns[-1]], axis = 'columns')
        self.valClass = np.array(self.valSet[self.valSet.columns[-1]])
        self.valSet = self.valSet.drop([self.valSet.columns[-1]], axis = 'columns')
        self.hiddenLayer = prototypeSet.drop([prototypeSet.columns[-1]], axis = 'columns')
        
        # add bias
        self.bias = np.random.randn(numberOfClasses, 1)
        self.updateBias = [0]
            
        # initialize weights random between [-1, 1]
        self.m = len(self.hiddenLayer)
        self.weights = np.random.uniform(-1, 1, size = (self.numOutNodes, self.m))
        self.updateWeights = [[0] * self.m for i in range(self.numOutNodes)]
        
        # learning rate 
        if self.classification:
            self.alpha = 3
        else:
            self.alpha = 0.1
                
    # train algorithm based on training set        
    def train(self):
        # repeat until convergence or max iterations
        result_list = []
        epoch_list = []
        epoch = 0
        while epoch < 100:
            print('epoch: ', epoch)
            # feedforward for each data point
            i = 0
            for node in self.inputLayer.values:
                nodeActivation = self.feedforward(node)
                # convert expected output to vector
                expected = self.convertOutput(self.inputClass[i])
                # train using backprop
                deltaBias, deltaWeights = Backprop.Backprop().backprop(np.reshape(nodeActivation, 
                                                           (len(nodeActivation), 1)), np.reshape(expected, (len(expected), 1)), [self.bias], [self.weights], self.backpropLayers)
                
                # store bias and weight changes
                self.updateBias += deltaBias[0]
                self.updateWeights += deltaWeights[0]
                i += 1
 
            # update weight matrix after iterating through training set
            self.weightUpdate(self.updateBias, self.updateWeights)
            
            # test with validation set
            predicted = self.test(self.valSet.values)
            result_list.append(self.calcLoss(self.valClass, predicted))
            epoch_list.append(epoch)
            print('validation loss: ', result_list[epoch])
            
            # reset updates to weight and bias back to zero
            self.updateWeights = [[0] * self.m for i in range(self.numOutNodes)]
            self.updateBias = [0]
            epoch += 1
        return epoch_list, result_list

    # feedforward one input node
    def feedforward(self, inputNode):
        nodeActivation = []
        # calculate results from gaussian activation function
        for hiddNode in self.hiddenLayer.values:
            result = self.gaussianActivation(inputNode, hiddNode)
            nodeActivation.append(result)
        nodeActivation = np.array(nodeActivation)
        nodeActivation.shape = (len(nodeActivation), 1)
        return nodeActivation
    
    # gaussian activation of input and hidden node
    def gaussianActivation(self, inputNode, hiddenNode):
            # get distance between input and hidden node
            d = self.metrics.euclideanDistance(inputNode, hiddenNode, len(inputNode))
            # calculate activation using gaussian rbf
            activationResult = exp(-(d**2))
            return activationResult  
    
    # get expected class or regression value for output node
    def convertOutput(self, output):
        if self.classification:
            expected = np.zeros((self.numOutNodes), dtype = int)
            expected[output] = 1
        else:
            expected = [output]
        return expected
      
    # update weights based on backprop results
    def weightUpdate(self, deltaBias, deltaWeights):        
        # average changes
        averageWeight = deltaWeights/len(self.inputLayer)
        averageBias = deltaBias/len(self.inputLayer)
        # weight with learning rate
        self.weights -= self.alpha*averageWeight  
        self.bias -= self.alpha*averageBias        
        
    # calculate loss
    def calcLoss(self, expected, predicted):
        if(self.classification):
            accuracy, precision, recall = self.metrics.confusion_matrix(np.array(expected), predicted)
            return accuracy
        else:
            loss = self.metrics.RootMeanSquareError(expected, predicted)
            return loss
    
    # test performance
    # drop class column before sending in test set from main
    def test(self, testingSet):
        predicted = []
        # get predictions
        for node in testingSet:
            activation = self.feedforward(node)
            currentPredicted = self.calcOutput(activation)
            predicted.append(currentPredicted)
         
        return predicted
     
    # calculate output layer
    def calcOutput(self, nodeActivation):
        # use matrix dot product
        output = np.dot(self.weights, nodeActivation) + self.bias
        self.outputLayer = self.metrics.sigmoid(output)

        # get prediction
        predicted = self.predict()
        return predicted
    
    # get predicted regression value or highest probability class
    def predict(self):
        if(self.classification):
            predicted = self.outputLayer.argmax()
        else:
            predicted = self.outputLayer[0]
        return predicted
    