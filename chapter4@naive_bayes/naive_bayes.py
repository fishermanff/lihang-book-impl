"""
@author chenjianfeng
@date 2018.01
""" 

from __future__ import print_function
import numpy as np

class NaiveBayes(object):
	def __init__(self, data_space, label_set, smooth=1.0):
		self.data_space = data_space
		self.label_set = label_set
		self.class_num = len(label_set)
		self.smooth = smooth
		self.condi_prob = {}
		self.priori_prob = {}

	def train(self, data, labels):
		m, n = np.shape(data)
		label_counts = {}
		for i in labels:
			if label_counts.has_key(i):
				label_counts[i] += 1.0
			else:
				label_counts[i] = 1.0
		for k in label_counts.keys():
			self.priori_prob[k] = (label_counts[k] + self.smooth) / (len(labels) + self.class_num * self.smooth)
			self.condi_prob[k] = {}
		for i, feature in enumerate(data):
			for j, val in enumerate(feature):
				if self.condi_prob[labels[i]].has_key((j,val)):
					self.condi_prob[labels[i]][j,val] += 1.0
				else:
					self.condi_prob[labels[i]][j,val] = 1.0
		for k in self.condi_prob.keys():
			for tk in self.condi_prob[k].keys():
				self.condi_prob[k][tk] = (self.condi_prob[k][tk] + self.smooth) / \
											(label_counts[k] + self.data_space[tk[0]] * self.smooth)
		print(self.priori_prob)

	def predict(self, X):
		prob_list = []
		for k in self.label_set:
			prob = self.priori_prob[k]
			for j, val in enumerate(X):
				prob = prob * self.condi_prob[k][j,val]
			prob_list.append((k, prob))
		return sorted(prob_list, key=lambda x:x[1], reverse=True)
		

if __name__ == '__main__':
	data = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
			[2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
			[3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
	labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
	data_space = [3, 3]		# number of available values of each dimension
	label_set = [1, -1]
	nb_classifier = NaiveBayes(data_space, label_set, smooth=1.0)
	nb_classifier.train(data, labels)
	test_data = [2, 'S']
	prob_out = nb_classifier.predict(test_data)
	print("prob_out:", prob_out)
	print("%s belongs to class %s." %(test_data, prob_out[0][0]))
