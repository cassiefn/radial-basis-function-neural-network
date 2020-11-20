# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:08:38 2019

@author: asadc
"""
'''
Class main. Calls the MLP and RBF for different datasets with
different network models and prints out the performance measures
'''

import LoadDataset as ld
import RBF
import Metrics
import pandas as pd
from graphs import plot_graph

class main:
        
        def __init__(self):
                #define variable to use inside class which may need tuning
                self.splitlength = 0.75  
                self.layers = [0, 1, 2]
                self.RBF_type = ['enn_', 'kmeans_', 'pam_']
                self.alldataset = ld.LoadDataset().load_data()          #load all dataset
                #check dataset is classification
                self.IsClassificationDict = ld.LoadDataset().IsClassificationDict() 
                #define dataframe to store all the results
                self.allresults = pd.DataFrame(columns=['dataset', 'isClassification', 'hiddenLayers', 'method',
                                                        'accuracy', 'precision', 'recall', 'RMSE'])
        
        def main(self):
                for dataset in self.alldataset:         #for each dataset call each algorithm
                        print('current dataset ::: {0} \n'.format(dataset))
                        data = self.alldataset.get(dataset)
                        isClassification = self.IsClassificationDict.get(dataset)
                        trainset, testset = self.testtrainsplit(data, self.splitlength)
                        len_output_neuron = 1
                        if isClassification: len_output_neuron = len(data[data.columns[-1]].unique())
                        #RUN RBF 
                        for type_ in self.RBF_type:
                                if isClassification:                                        
                                        predicted, labels = self.run_RBF(dataset, data, trainset, testset, isClassification,
                                                                 len_output_neuron, type_)
                                else:
                                        if type_ == 'enn_': continue
                                        predicted, labels = self.run_RBF(dataset, data, trainset, testset, isClassification,
                                                                 len_output_neuron, type_)
                                self.performance_measure(predicted, labels, dataset, isClassification, 1 , 'RBF_' + type_)
                        
                return self.allresults
                        
                        
                        
        def testtrainsplit(self, data, foldlen):
                data = data.sample(frac=1)                   #randomize the rows to avoid sorted data
                testlen = int(len(data) * foldlen)           #split according to fold lenth
                testset = data[testlen:]
                trainset = data[:testlen]
                return trainset, testset
        
        def run_RBF(self, key, dataset, train, test, isClassification, output_neuron, type_):
                prot = ld.LoadDataset().getRBFhiddenLayer(key, type_)
                rbf = RBF.RBF(train, prot, isClassification, output_neuron)
                epoch, result = rbf.train()
                plot_graph(key+' with RBF type '+ type_, epoch, result)
                x_test, test_label = ld.LoadDataset().get_neural_net_input_shape(dataset, test, isClassification)
                predicted = rbf.test(x_test)
                return predicted, test.iloc[:, -1]
                
        def performance_measure(self, predicted, labels, dataset, isClassification, h, method):
                #evaluate confusion metrix or root mean square error based on dtaset
                mtrx = Metrics.Metrics()
                if (isClassification):
                        acc, prec, recall = mtrx.confusion_matrix(labels.values, predicted)
                        self.update_result(dataset, isClassification, h, method, acc, prec, recall, 0)
                         
                else:
                        rmse = mtrx.RootMeanSquareError(labels.values, predicted)
                        self.update_result(dataset, isClassification, h, method, 0, 0, 0, rmse)
        
        def update_result(self, dataset, isClassification, h, method, acc, prec, recall, rmse):
                #store result in a dataframe.
                self.allresults = self.allresults.append({'dataset': dataset, 'isClassification': isClassification,
                                                'hiddenLayers': h, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'RMSE': rmse}, ignore_index=True)
        
        
        
        
        
        
results = main().main()
results.to_csv('results.csv')
                