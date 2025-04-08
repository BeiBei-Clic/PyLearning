from tkinter.scrolledtext import example

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

from numpy.ma.extras import unique


def createDataSet():
    dataSet=[
        [0,0,0,0,'no'],
        [0,0,0,1,'no'],
        [0,1,0,1,'yes'],
        [0,1,1,0,'yes'],
        [0,0,0,0,'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'yes'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
    ]
    labels=['F1-AGE','F2-WORK','F3-HOME','F4-LOAN']
    return dataSet,labels

def createTree(dataset,labels,featLabels):

    #判断是否抵达叶子结点
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):#判断叶子结点是不是已经包含了全部值
        return  classList[0]
    if len(dataset[0])==1:#只剩下一个标签，遍历完了
        return majorityCnt(classList)

    bestFeat=chooseBestFeatureToSplit(dataset)#选择最好的特征
    bestFeatLabel=labels[bestFeat]#找到当前的label值
    featLabels.append(bestFeatLabel)
    myTree={bestFeatLabel:{}}
    del labels[bestFeat]
    featValue=[example[bestFeat] for example in dataset]
    uniqueVals=set(featValue)
    for value in uniqueVals:
        sublabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataset,bestFeat,value),sublabels,featLabels)
    return myTree

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedclassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]

def chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1
    baseEntropy=calcShannoEnt(dataset)#计算旧结点的熵值
    bestInfoGain=0#最好的信息增益
    bestFeature=-1
    for i in range(numFeatures):#遍历每一列（每一个特征）
        featList=[example[i] for example in dataset]
        uniueVals=set(featList)
        newEntropy=0
        for val in uniueVals:
            subDataSet=splitDataSet(dataset,i,val)
            prob=len(subDataSet)/float(len(dataset))
            newEntropy+=prob*calcShannoEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestFeature=infoGain
            bestFeature=i
    return bestFeature


def splitDataSet(dataset,axis,val):
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis]==val:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet




def calcShannoEnt(dataset):
    numexamples=len(dataset)
    labelCounts={}
    for featVec in dataset:
        currentlabel=featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel]=0
        labelCounts[currentlabel] += 1
    shannonEnt=0
    for key in labelCounts:
        prop=float(labelCounts[key])/numexamples
        shannonEnt-=prop*log(prop,2)
    return shannonEnt


if __name__=='__main__':
    dataset,labels=createDataSet()
    featLabels=[]
    myTree=createTree(dataset, labels, featLabels)