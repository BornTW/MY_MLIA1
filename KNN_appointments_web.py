# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:29:04 2019

@author: 吕曌
"""
from numpy import *
import matplotlib
import operator
import matplotlib.pyplot as plt

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

"""
第一步、1.准备数据：从文本文件中解析数据
"""
#将待处理的数据改变为分类器可以接受的格式
#该函数的输入为文件名字符串，输出为训练样本矩阵和标签向量
def file2matrix(filename):
    fr = open(filename)
    #readlines()函数一次读取整个文件，readlines() 自动将文件内容分析成一个行的列表，
    #该列表可以由 Python 的 for ... in ... 结构进行处理。
    arrayOLines = fr.readlines()
    #len() 返回字符串、列表、字典、元组等长度。
    #得到文件的行数
    numberOfLines = len(arrayOLines)
    #zeros函数 例:zeros((3,4)),创建3行4列以0填充的矩阵
    #创建以0填充的Numpy 矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #strip()函数 s.strip(rm) 删除s字符串中开头、结尾处，位于 rm删除序列的字符
        #当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        line = line.strip()  # 截取所有的回车符
        #split()：拆分字符串。通过指定分隔符对字符串进行切片，
        #并返回分割后的字符串列表（list）
        listFromLine = line.split('\t')  #解析文件数据到列表
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
        
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')        
        
'''
#测试
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
print(" datingDataMat \n " , datingDataMat," \n")
print(" datingLabels \n" , datingLabels[0:20])

     
fig = plt.figure()  
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()            
 '''        
        
 #该函数可以将数字特征值转化为0到1的区间
def autoNorm(dataSet):
    #a.min()返回的就是a中所有元素的最小值
    #a.min(0)返回的就是a的每列最小值
    #a.min(1)返回的是a的每行最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #生成一个和数据集相同大小的矩阵
    normDataSet = zeros(shape(dataSet))
    #获取数据集行数
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    #特征值相除
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#测试函数
normMat, ranges, minVals = autoNorm(datingDataMat)
'''
print("normMat: \n", normMat,"\n ")
print("ranges: \n", ranges," \n ")
print("minVals: \n", minVals)
'''
        
#测试分类器的效果函数
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获取数据集的行数
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # "\" 换行符
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                       datingLabels[numTestVecs:m], 20)
        print("the classsifier came back with: %d, the real answer is: %d"\
                       %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount +=1.0
    print("the total error rate is:%f" % (errorCount/float(numTestVecs)))

#datingClassTest()   #测试分类器的正确率 测试算法        
       
#使用算法
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    # \ 为换行符
    percentTats = float(input(\
                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input ("liters of ice creamm consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-\
                       minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ",\
            resultList[classifierResult - 1])
classifyPerson()    








 
        