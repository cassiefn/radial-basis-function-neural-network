# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:21:36 2019

@author: asadc
"""

'''
LoadDataset class loads all the data from the csv files
and converts to datafiles. PreprocessingData() handles
the categorical values and class reorder. NormalizeData() normalizes
all of the datasets as we are using sigomoid function. There is another
function to format the input to feed in the neural net model.
'''

import pandas as pd
import numpy as np

class LoadDataset:
        def __init__(self):
                self.directory = 'dataset/'
                #self.datafiles = ['abalone.data','car.data','segmentation.data', 'machine.data',
                                 #'forestfires.data', 'winequality-red.csv', 'winequality-white.csv']
                self.datafiles = ['segmentation.data', 'forestfires.data']
                self.alldataset = {}
                
        def load_data(self):
                for files in self.datafiles:       
                        #read each datafiles
                        data = pd.read_csv(self.directory + files)
                        #give filename without extension as dict key for each dataset
                        key = files.split('.')[0]
                        self.alldataset[key] = self.PreprocessingData(key, data)
                return self.normalize_data()
                
        def PreprocessingData(self, key, data):
                if key == 'abalone':
                        data = data.drop(['Sex'], axis= 1)
                        classes = data[data.columns[-1]].values
                        classes = classes - 1
                        data[data.columns[-1]] = classes
                        data = data.replace({data.columns[-1] : {28:27}})
                elif key == 'forestfires':
                        data = data.drop(['month', 'day'], axis= 1)
                elif key == 'machine':
                        data = data.drop(['Vendor name', 'Model name', 'ERP'], axis= 1)
                elif key == 'segmentation':
                        data = data.replace({'CLASS': {'BRICKFACE': 0, 'SKY': 1, 'FOLIAGE': 2, 'CEMENT': 3,
                                                       'WINDOW': 4, 'GRASS': 5, 'PATH': 6 }})
                        class_d = data['CLASS']
                        data = data.drop(['CLASS'], axis= 1)
                        data['CLASS'] = class_d
                elif key == 'car':
                        data = data.replace({'low': 0, 'med': 1, 'high': 2, 'vhigh':3, '5more': 5,
                                             'more': 5, 'small': 0, 'big': 2, 'unacc': 0, 'acc': 1, 
                                             'good': 2, 'vgood': 3})
                        data[['doors', 'persons']] = data[['doors', 'persons']].astype(int)
                return data
        
        def normalize_data(self):
                #normalize dataset points with min-max normalization
                for key in self.alldataset:
                        data = self.alldataset.get(key)
                        isClassification = self.IsClassificationDict().get(key)
                        
                        self.alldataset[key] = self.normalize(data, isClassification)
                return self.alldataset
        
        def normalize(self, data, isClassification):
                #is dataset is classification don't normalize the class output
                #otherwise for regression normalize the prediction output.
                if isClassification:    cols = data.columns[:-1] 
                else:   cols = data.columns
                for col in cols:
                    col_values = data[col].values
                    value_min = min(col_values)
                    value_max = max(col_values)
                    data[col] = (col_values - value_min) / (value_max - value_min)
                data = data.fillna(0)
                return data
        
        def get_neural_net_input_shape(self, data_all,  dataset, isClassification = True):
                #get data for neural net format with a unit vector for class output label
                #containing 1 for that class and 0's for other class. For regression we have
                #only one output layer with the actual value
                data = list()
                label = list()
                class_len = len(data_all[data_all.columns[-1]].unique())
                for index, row in dataset.iterrows():
                        if isClassification:                                
                                row_label = int(row[dataset.columns[-1]])
                                unit_vec = np.zeros((class_len, 1))
                                unit_vec[row_label] = 1
                                label.append(unit_vec)
                        else:
                                label.append(row[dataset.columns[-1]])
                        data.append(np.reshape(np.asarray(row[dataset.columns[:-1]]),
                                               (len(dataset.columns[:-1]), 1)))
                return data, label
                        
        
        def IsClassificationDict(self):
                #return if dataset is classification or regression.
                return {'abalone': True, 'car': True, 'segmentation': True, 'machine': False,
                        'forestfires': False, 'winequality-red': False, 'winequality-white': False} 
                
        def get1sthiddenlayernode(self, key):
                #define hidden layer 1 node number based on dataset and datapoints
                #and tuned to get best performance.
                dict_list = {'abalone': 60, 'car': 30, 'segmentation': 15, 'machine': 30,
                        'forestfires': 30, 'winequality-red': 20, 'winequality-white': 60}
                return dict_list.get(key)
        
        def get2ndhiddenlayernode(self, key):
                #define node number for 2 layers for each dataset and tune to get 
                #best performance.
                dict_list = {'abalone': [20,20], 'car': [15,15], 'segmentation': [8, 8], 'machine': [16,16] ,
                        'forestfires': [15,15], 'winequality-red': [8,8], 'winequality-white': [16, 16]}
                return dict_list.get(key)
        
        def getRBFhiddenLayer(self, key, type_):
                #get hidden layer for RBF already stored from the result of project 2
                #type_ is enn/Kmenas/PAM reduced datapoints
                #key is the dataset
                prot = pd.read_csv('reduced_datasets/' + type_ + key + '.csv')
                isClassification = self.IsClassificationDict().get(key)
                prot = self.normalize(prot, isClassification)
                return prot
                