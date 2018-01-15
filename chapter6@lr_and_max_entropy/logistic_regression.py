"""
@author chenjianfeng
@date 2018.01
""" 

from __future__ import print_function
import numpy as np

class LogisticRegressionClassifier(object):
	def __init__(self, dim):
		self.theta = np.random.rand(dim+1)

	def sigmoid(self, X):
		return 1.0 / (1.0 + np.exp(-X))

	def calcGradient(self, X, y):
		return ((self.sigmoid(self.theta.dot(X.T)) - y) * X.T).T

	def train(self, data, labels, maxIter=20, alpha=0.02):
		m, n = np.shape(data)
		if (n) != len(self.theta):
			raise Exception("Theta dimension is not equal to input data dimension.")
		it = 0
		while it < maxIter:
			grad = self.calcGradient(data, labels)
			self.theta = self.theta - alpha * 1.0/m * sum(grad)
			print("Iteration %d finished." %(it))
			print("avgGrad:",sum(grad)/len(grad))
			it += 1

	def predict(self, X):
		prob = self.sigmoid(self.theta.dot(X.T))
		y_pred = [1 if p>0.5 else 0 for p in prob ]
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
			labels.append(0)
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
	data, labels = genDataset(500, dim)
	trainData, trainLabels, testData, testLabels = splitDataset(data, labels, 0.8)
	print("trainData:", trainData)
	print("trainLabels:", trainLabels)
	lrClassifier = LogisticRegressionClassifier(dim)
	lrClassifier.train(trainData, trainLabels, maxIter=60, alpha=0.02)
	test_pred = lrClassifier.predict(testData)
	for i in range(len(testData)):
		print("actual:%d, predict:%d" %(testLabels[i], test_pred[i]))
	result = [1 if test_pred[i]==testLabels[i] else 0 for i in range(len(test_pred))]
	print("Test set accuracy is %.2f%%" %(sum(result)*100/float(len(result))))


