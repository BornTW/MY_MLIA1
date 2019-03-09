# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:28:00 2019

@author: 吕曌
"""
from numpy import *
import operator
from os import listdir

"""
手写数据集 准备数据：将图像转换为测试向量
"""
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

#testVector = img2vector('testDigits/0_13.txt')
#print(testVector[0,0:22])

def classify0(inX,dataSet,labels,k):
    #获取训练数据集的行数
    dataSetSize=dataSet.shape[0]
    #---------------欧氏距离计算-----------------
    #各个函数均是以矩阵形式保存
    #tile():inX沿各个维度的复制次数
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    #.sum()运行加函数，参数axis=1表示矩阵每一行的各个值相加和
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #--------------------------------------------
    #获取排序（有小到大）后的距离值的索引（序号）
    sortedDistIndicies=distances.argsort()
    #字典，键值对，结构类似于hash表
    classCount={}
    for i in range(k):
        #获取该索引对应的训练样本的标签
        voteIlabel=labels[sortedDistIndicies[i]]
        #累加几类标签出现的次数，构成键值对key/values并存于classCount中
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #将字典列表中按照第二列，也就是次数标签，反序排序（由大到小排序）
    sortedClassCount=sorted(classCount.items(),
     key=operator.itemgetter(1),reverse=True)
    #返回第一个元素（最高频率）标签key
    return sortedClassCount[0][0]

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,\
                           trainingMat, hwLabels, 3 )
        print("the classifier came back with: %d, the real answer is: %d"\
                  % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n the total number of errors is:%d" % errorCount)
    print("\n the total error rate is: %f" % (errorCount/float(mTest)))

handwritingClassTest()





