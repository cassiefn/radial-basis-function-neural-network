# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:15:47 2019

@author: asadc
"""
'''
Metrics class holds the general methods used by
main, Backprop, and RBF class
'''
import numpy as np

class Metrics:

        def euclideanDistance(self, x, y, length):
                distance = 0
                for i in range(length):
                    distance += np.power((x[i] - y[i]),(2)) 
                distance = np.sqrt(distance)
                return distance
                
        def confusion_matrix(self, y_test, y_pred):
                #set to 0.0000001 to protect division by zero. Soybean dataset is very small 
                #and with 10 fold there are only 3 to 5 values which may produce zero division.
                TP = 0.0000001
                TN = 0.0000001
                FP = 0.0000001
                FN = 0.0000001
                #find unique class label for test and pred
                test_label = np.unique(np.array(y_test))
                pred_label = np.unique(np.array(y_pred))
                
                #work with max labels from test and pred
                if len(test_label) >= len(pred_label):
                   labels = test_label
                else:
                   labels = pred_label
                #find the confusion matrix values   
                for x in labels:
                    for y in range(len(y_test)):
                        if y_test[y] == y_pred[y] == x:                 
                            TP += 1
                        if y_pred[y]== x and y_test[y] != y_pred[y]:
                            FP += 1
                        if y_test[y] == y_pred[y] != x:
                            TN += 1
                        if y_pred[y] != x and y_test[y] != y_pred[y]:
                            FN += 1
                            
                accuracy = ((TP + TN)/ (TP + TN + FP + FN))
                precision = TP/(FP + TP)
                recall = TP/(FN + TP)
                return accuracy, precision, recall    
        
        def RootMeanSquareError(self, y_test, y_pred):
                return np.sqrt(np.mean((y_pred-y_test)**2))
        
        ###useful functions for backprop, and RBF
        def cost_derivative(self, predicted, actual):
                return (predicted - actual)        
        
        def sigmoid(self, z):
                return 1.0/(1.0+np.exp(-z)) 
        
        def sigmoid_derivative(self, z):
                return self.sigmoid(z)*(1 - self.sigmoid(z))