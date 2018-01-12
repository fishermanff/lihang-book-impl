"""
@author chenjianfeng
@date 2018.01
""" 

# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib  
import matplotlib.pyplot as plt  
from smo import SMO
from sparseVector import SparseVector
from kernel import Kernel

class SVM(object):
    def __init__(self, dataset, labels, C=1.0, tolerance=1e-4, maxIter=10000, kernelTup=('linear',)):
    	self.dataset = dataset
    	self.labels = labels
        self.C = C
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.kern = Kernel(kernelTup=kernelTup)
        self.m = len(dataset)
        self.alphaArr = np.zeros((self.m,1))
        self.b = 0

    def train(self):
    	smoAlgor = SMO(dataset = self.dataset, 
    					 labels = self.labels, 
    					 C = self.C, 
    					 tolerance = self.tolerance, 
    					 maxIter = self.maxIter,
    					 kernel = self.kern)
    	smoAlgor.mainRoutine()
    	self.alphaArr = smoAlgor.alphaArr
    	self.b = smoAlgor.b

    def predict(self, X):
    	result = []
    	m = len(self.dataset)
    	for v in X:
    		K = np.array([self.kern.kernel(xi,v) for xi in self.dataset])
    		val = sum(self.labels * self.alphaArr * K) + self.b
    		result.append(1 if val > 0 else -1)
    	return result

    def show(self):
		class1_x1 = []; class1_x2 = []
		class2_x1 = []; class2_x2 = []
		class1sv_x1 = []; class1sv_x2 = []
		class2sv_x1 = []; class2sv_x2 = []
		for i in range(len(self.dataset)):
			if self.labels[i] == 1:
				if self.alphaArr[i] > 0:
					class1sv_x1.append(self.dataset[i].vals[0] if 1 in self.dataset[i].ids else 0.)
					class1sv_x2.append(self.dataset[i].vals[-1] if 2 in self.dataset[i].ids else 0.)
				else:
					class1_x1.append(self.dataset[i].vals[0] if 1 in self.dataset[i].ids else 0.)
					class1_x2.append(self.dataset[i].vals[-1] if 2 in self.dataset[i].ids else 0.)
			else:
				if self.alphaArr[i] > 0:
					class2sv_x1.append(self.dataset[i].vals[0] if 1 in self.dataset[i].ids else 0.)
					class2sv_x2.append(self.dataset[i].vals[-1] if 2 in self.dataset[i].ids else 0.)
				else:					
					class2_x1.append(self.dataset[i].vals[0] if 1 in self.dataset[i].ids else 0.)
					class2_x2.append(self.dataset[i].vals[-1] if 2 in self.dataset[i].ids else 0.)
		fig = plt.figure()
		plt.scatter(class1_x1, class1_x2, marker='o', c='g', label='class1(1)', s=40)
		plt.scatter(class1sv_x1, class1sv_x2, marker='d', c='g', label='sv of class1', s=40)
		plt.scatter(class2_x1, class2_x2, marker='o', c='r', label='class2(-1)', s=40)
		plt.scatter(class2sv_x1, class2sv_x2, marker='x', c='r', label='sv of class2', s=40)
		if self.kern.kernelTup[0] == 'linear':
			w = np.ones(2)
			x1 = [d.vals[0] if 1 in d.ids else 0. for d in self.dataset]
			x2 = [d.vals[-1] if 2 in d.ids else 0. for d in self.dataset]
			w[0] = sum(self.alphaArr * self.labels * x1)
			w[1] = sum(self.alphaArr * self.labels * x2)
			x1_min = min(x1)
			x1_max = max(x1)
			x2_x1_min = float(-self.b - w[0] * x1_min) / w[1]
			x2_x1_max = float(-self.b - w[1] * x1_max) / w[0]
			plt.plot([x1_min, x1_max], [x2_x1_min, x2_x1_max], '-b')
		plt.xlabel('x1'); plt.ylabel('x2'); plt.legend() 
		plt.show()

def readDataset(path):
	data = []
	labels = []
	with open(path,'r') as fin:
		for line in fin.readlines():
			line_split = line.split()
			labels.append(int(line_split[0]))
			ids = []
			vals = []
			for i in range(1,len(line_split)):
				ids.append(int(line_split[i].split(':')[0]))
				vals.append(float(line_split[i].split(':')[1]))
			data.append(SparseVector(ids,vals))
	return data, labels

def genDataset(size):
	data = []
	labels = []
	for i in range(size):
		if np.random.rand() < 0.5:
			labels.append(1)
			data.append(SparseVector([1,2],[np.random.normal(1,0.5), np.random.normal(1,0.5)]))
		else:
			labels.append(-1)
			data.append(SparseVector([1,2],[np.random.normal(-1,0.5), np.random.normal(-1,0.5)]))
	return data, labels		

def splitDataset(data, labels, ratio):
	trainData = []; trainLabels = []
	testData = []; testLabels = []
	for i in range(len(data)):
		if np.random.rand() < ratio:
			trainData.append(data[i])
			trainLabels.append(labels[i])
		else:
			testData.append(data[i])
			testLabels.append(labels[i])
	return trainData, trainLabels, testData, testLabels

if __name__ == '__main__':
	np.random.seed(10)
	# data, labels = readDataset('fourclass_scale.txt')
	data, labels = genDataset(200)
	trainData, trainLabels, testData, testLabels = splitDataset(data, labels, 0.8)
	print("trainData num: %d" %len(trainData))
	print("trainLabels num: %d" %len(trainLabels))
	print("testData num: %d" %len(testData))
	print("testLabels num: %d" %len(testLabels))
	svm = SVM(trainData, trainLabels, C=0.9, tolerance=1e-4, maxIter=600, kernelTup=('linear',))
	svm.train()
	preds = svm.predict(testData)
	rt = [1 if preds[i]==testLabels[i] else 0 for i in range(len(preds))]
	for i in range(len(testData)):
		print("actual:%d, predict:%d" %(testLabels[i], preds[i]))
	print("Classification accuracy: %.2f%%" %(sum(rt)*100/float(len(rt))))
	svm.show()
