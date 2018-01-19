"""
@author chenjianfeng
@date 2018.01
""" 

from __future__ import print_function
import numpy as np
import copy
from basic_classifier import BasicClassifier

class AdaBoost(object):
	def __init__(self, data=None, labels=None, basic_clf=None, v_list=None, symbol=None, max_num=3):
		self.data = data
		self.labels = labels
		self.basic_clf = basic_clf
		self.v_list = v_list
		self.max_num = max_num
		self.n = len(data)
		self.m = 0
		self.weights = np.array([1.0/self.n] * self.n)
		self.alpha = []
		self.func = []

	def train(self):
		it = 0
		while it < self.max_num:
			# solve best basic classifier
			err_min = 1.
			curr_alpha = 0.
			v_best = self.v_list[0]
			y_pred_best = []
			symbol_best = ''
			for symbol in self.basic_clf.symbol_set:
				for v in self.v_list:
					self.basic_clf.symbol = symbol
					self.basic_clf.v = v
					y_pred = self.basic_clf.predict(self.data)
					err = self.calcWeightedErrorRate(y_pred)
					if err < err_min:
						curr_alpha = 0.5 * np.log((1 - err) / err)
						err_min = err
						v_best = v
						symbol_best = symbol
						y_pred_best = y_pred
			self.alpha.append(curr_alpha)
			self.func.append(copy.deepcopy(self.basic_clf.setV(v_best).setSymbol(symbol_best)))
			print("The training error rate produced by G_%d(x) is %.4f." %(it+1, err_min))

			# update weights
			factor = [- curr_alpha * yi * gi for yi, gi in zip(self.labels, y_pred_best)]
			den = sum(self.weights * np.exp(factor))
			self.weights = self.weights * np.exp(factor) / den

			it += 1

	def calcWeightedErrorRate(self, y_pred):
		err = [1 if y_pred[i] != self.labels[i] else 0 for i in range(self.n)]
		return sum(self.weights * err)

if __name__ == '__main__':
	X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
	v_list = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
	adaboost_clf = AdaBoost(data=X, labels=y, basic_clf=BasicClassifier(0), v_list=v_list, max_num=3)
	adaboost_clf.train()
	print("The final boosted classifier is:")
	print("	f(x) = sign[%.4f*G_1(x)+%.4f*G_2(x)+%.4f*G_3(x)]" %tuple(adaboost_clf.alpha))
	print("where,")
	print("	G_1(x) = 1 if x %s %.1f else -1" %(adaboost_clf.func[0].symbol, adaboost_clf.func[0].v))
	print("	G_2(x) = 1 if x %s %.1f else -1" %(adaboost_clf.func[1].symbol, adaboost_clf.func[1].v))
	print("	G_3(x) = 1 if x %s %.1f else -1" %(adaboost_clf.func[2].symbol, adaboost_clf.func[2].v))




