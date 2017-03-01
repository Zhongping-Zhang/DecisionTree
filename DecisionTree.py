# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 08:54:50 2017

@author: Zhongping Zhang
"""
import numpy as np
from math import log

def array_convert_to_list(dataSet):
    dataSet2 = list(dataSet)
    length = len(dataSet2)
    for i in range(length):
        dataSet2[i] = list(np.round(dataSet2[i]))
    return dataSet2

def Calculate_Shannon_Entropy(dataSet):  
    numEntries = len(dataSet)  
    labelCounts = {}  
    for featVec in dataSet:      #create the dictionary for all of the data  
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    shannonEnt = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        shannonEnt -= prob*log(prob,2) #get the log value  
    return shannonEnt  

def Gini_Index(dataSet):
    numEntries = len(dataSet)  
    labelCounts = {}  
    for featVec in dataSet:      #create the dictionary for all of the data  
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    #print(labelCounts)
    GiniIndex = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        GiniIndex += prob**2 #get the log value  
    GiniIndex = 1-GiniIndex
    #print('GiniIndex is'+str(GiniIndex))
    return GiniIndex  

def Misclassification_Error(dataSet):
    numEntries = len(dataSet)  
    labelCounts = {}  
    for featVec in dataSet:      #create the dictionary for all of the data  
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    error = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries 
        #print(prob)
        if (prob>error):
            error = prob
        #print(error)
    error = 1-error
    return error  


def Classify_Dataset(dataSet, position, value):  
    #axis: the position of classification element
    #value: feature of classification element we want to extract
    one_category_dataset = []  
    for featureVector in dataSet:  
        if featureVector[position] == value:      
            #extract vector which belongs to one category  
            reducedFeatVec = featureVector[:position]  #A[:axis] means A[0:axis]
            reducedFeatVec.extend(featureVector[position+1:]) #A[axis:] means A[axis:end]
#This two lines means assign values of the featureVector except classification 
#position's value to reducedFeatVec list
            one_category_dataset.append(reducedFeatVec)
            #need to use append to split different featureVectors
    return one_category_dataset

def chooseBestFeatureToSplit(dataSet,method):
    numFeatures = len(dataSet[0])#number of features of one category dataset
    if method == 'Entropy':
        Total_Entropy = Calculate_Shannon_Entropy(dataSet)  
        bestInfoGain = 0.0; bestFeature = 0 #initialize information gain 
        for i in range(numFeatures-1):  #numFeatures = 2 here, i = 0,1
            Feature_Extract = [example[i] for example in dataSet]  #when i =0, featList = [1,1,1,0,0]
            Different_Features = set(Feature_Extract)  #extract different values of featList
            sum_subEntropy = 0.0  
            for value in Different_Features:
#Different_Features means the number of categories of a classification method   
                sub_dataSet = Classify_Dataset(dataSet, i , value)  
                prob = len(sub_dataSet)/float(len(dataSet))  
                sum_subEntropy +=prob * Calculate_Shannon_Entropy(sub_dataSet)  
                Information_Gain = Total_Entropy - sum_subEntropy
                if(Information_Gain > bestInfoGain):  
                    bestInfoGain = Information_Gain  
                    bestFeature = i  
        return bestFeature  
    if method == 'Gini Index':
        Total_Gini = Gini_Index(dataSet)  
        bestInfoGain = 0.0; bestFeature = 0 #initialize information gain 
        for i in range(numFeatures-1):  #numFeatures = 2 here, i = 0,1
            Feature_Extract = [example[i] for example in dataSet]  #when i =0, featList = [1,1,1,0,0]
            Different_Features = set(Feature_Extract)  #extract different values of featList
            sum_subGini = 0.0  
            for value in Different_Features:
#Different_Features means the number of categories of a classification method   
                sub_dataSet = Classify_Dataset(dataSet, i , value)  
                prob = len(sub_dataSet)/float(len(dataSet))  
                sum_subGini +=prob * Gini_Index(sub_dataSet)  
                Information_Gain = Total_Gini - sum_subGini
                if(Information_Gain > bestInfoGain):  
                    bestInfoGain = Information_Gain  
                    bestFeature = i  
        return bestFeature
    if method == 'Misclassification error':
        Total_error = Misclassification_Error(dataSet)  
        bestInfoGain = 0.0; bestFeature = 0 #initialize information gain 
        for i in range(numFeatures-1):  #numFeatures = 2 here, i = 0,1
            Feature_Extract = [example[i] for example in dataSet]  #when i =0, featList = [1,1,1,0,0]
            Different_Features = set(Feature_Extract)  #extract different values of featList
            sum_suberror = 0.0  
            for value in Different_Features:
#Different_Features means the number of categories of a classification method   
                sub_dataSet = Classify_Dataset(dataSet, i , value)  
                prob = len(sub_dataSet)/float(len(dataSet))  
                sum_suberror +=prob * Misclassification_Error(sub_dataSet)  
                Information_Gain = Total_error - sum_suberror
                if(Information_Gain > bestInfoGain):  
                    bestInfoGain = Information_Gain  
                    bestFeature = i  
        return bestFeature  

def majorityFeatureClassification(classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys():
            classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.items(),key = lambda item:item[1])
    feature  = sortedClassCount[len(classCount)-1][0]
    error = len(classList)-classCount[feature]
    return feature,error 

def createTree(dataSet, labels,method,num_attributes,depth_tree):   
    classList = [example[-1] for example in dataSet]  
    if classList.count(classList[0]) == len(classList):  
        return classList[0]  
    if (len(dataSet[0]) == 1):  
        a,error = majorityFeatureClassification(classList)
        print(error)
        return a
    if (len(dataSet[0]) == num_attributes + 1 - depth_tree):
        a,error = majorityFeatureClassification(classList)
        print(error)
        return a
    
    bestFeat = chooseBestFeatureToSplit(dataSet,method)
    bestFeatLabel = labels[bestFeat]  
    myTree = {bestFeatLabel:{}}  
    del(labels[bestFeat])  
    featValues = [example[bestFeat] for example in dataSet]  
    uniqueVals = set(featValues)  
    for value in uniqueVals:  
        subLabels = labels[:]  
        myTree[bestFeatLabel][value] = createTree(Classify_Dataset(dataSet, bestFeat, value), subLabels,method,num_attributes,depth_tree)  
    return myTree


def store(data):
    matrix = np.zeros((len(data),4))
    for i in range(len(data)):
        matrix[i] = data[i].split(',')[0:4]
    return matrix

# CreateData
def createDataSet():  
    dataSet = [[1,1,'yes'],[1,1, 'yes'],[1,0,'no'],[0,1,'no'], [0,1,'no']]  
    labels = ['no surfacing','flippers']  
    return dataSet, labels  

def Iris_DataSet():
    with open('iris.data','r') as f:
        total_data = f.read();
    
    raw_data = total_data.split('\n')
    raw_data = raw_data[0:150]
    raw_matrix = store(raw_data)
    raw_list = array_convert_to_list(raw_matrix)

    def append_list(raw_list):
        for i in range(50):
            c = 'setosa'
            raw_list[i].append(c)
        for i in range(50,100):
            c = 'versicolor'
            raw_list[i].append(c)
        for i in range(100,150):
            c = 'virginica'
            raw_list[i].append(c)
        return raw_list
    append_list = append_list(raw_list)
    labels = ['sepal length','sepal width','petal length','petal width']
    return append_list,labels 


