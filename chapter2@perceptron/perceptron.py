"""
@author chenjianfeng
@date 2018.02
""" 

from __future__ import print_function
import numpy as np

class Perceptron(object):
	def __init__(self, dim):
		self.theta = np.random.rand(dim+1)

	def train(self, data, labels, max_iter=20, alpha=0.02):
		m, n = np.shape(data)
		if n != len(self.theta):
			raise Exception("Theta dimension is not equal to input data dimension.")
		it = 0
		while it < max_iter:
			for (x,y) in zip(data,labels):
				if y*(self.theta.dot(x)) <= 0:
					self.theta += alpha * y * x
			it += 1

	def predict(self, X):
		output = self.theta.dot(X.T)
		y_pred = [1 if p>0 else -1 for p in output ]
		return y_pred

def genDataset(size, dim):
	data = []
	labels = []
	for i in range(size):
		if np.random.rand() < 0.5:
			labels.append(1)
			x = np.random.normal(1,1,dim)
			data.append(np.append(x,1))
		else:
			labels.append(-1)
			x = np.random.normal(-1,1,dim)
			data.append(np.append(x,1))
	return np.array(data), np.array(labels)		

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
	return np.array(trainData), np.array(trainLabels), np.array(testData), np.array(testLabels)

if __name__ == '__main__':
	np.random.seed(47)
	dim = 3
	data, labels = genDataset(200, dim)
	trainData, trainLabels, testData, testLabels = splitDataset(data, labels, 0.8)
	print("trainData:", trainData)
	print("trainLabels:", trainLabels)
	clf = Perceptron(dim)
	clf.train(trainData, trainLabels, max_iter=30, alpha=0.02)
	test_pred = clf.predict(testData)
	for i in range(len(testData)):
		print("actual:%d, predict:%d" %(testLabels[i], test_pred[i]))
	result = [1 if test_pred[i]==testLabels[i] else 0 for i in range(len(test_pred))]
	print("Test set accuracy is %.2f%%" %(sum(result)*100/float(len(result))))