# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:01:33 2017

@author: DELL
"""
import DecisionTree as tree
import treePlotter

DataSet,labels = tree.Iris_DataSet()
print('The error number is :')
myTree = tree.createTree(DataSet,labels,'Entropy',4,2)
#4 means number of attributes, 2 means depth of my tree
print('The training result is :')
print(myTree)
treePlotter.createPlot(myTree)


DataSet,labels = tree.Iris_DataSet()
print('The error number is :')
myTree = tree.createTree(DataSet,labels,'Gini Index',4,2)
#4 means number of attributes, 2 means depth of my tree
print('The training result is :')
print(myTree)
treePlotter.createPlot(myTree)

DataSet,labels = tree.Iris_DataSet()
print('The error number is :')
myTree = tree.createTree(DataSet,labels,'Misclassification error',4,2)
#4 means number of attributes, 2 means depth of my tree
print('The training result is :')
print(myTree)
treePlotter.createPlot(myTree)

'''
DataSet,labels = tree.Iris_DataSet()
myTree1 = tree.createTree(DataSet,labels,'Gini Index')
#print(myTree)
treePlotter.createPlot(myTree1)

DataSet,labels = tree.Iris_DataSet()
myTree2 = tree.createTree(DataSet,labels,'Gini Index')
#print(myTree)
treePlotter.createPlot(myTree2)
'''
