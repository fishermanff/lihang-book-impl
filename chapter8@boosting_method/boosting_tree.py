"""
@author chenjianfeng
@date 2018.01
""" 

from __future__ import print_function
import numpy as np
import copy
from basic_regressor import BasicRegressor

class BoostingTree(object):
	def __init__(self, X, y, split_point, max_num=6):
		self.X = X
		self.y = y
		self.split_point = split_point
		self.max_num = max_num
		self.residual = self.y
		self.func = []
		self.n = len(self.X)

	def train(self):
		it = 0
		while it < self.max_num:
			pred = np.zeros(self.n)
			for tree in self.func:
				pred += tree.predict(self.X)
			self.residual = self.y - pred
			best_split, c1, c2 = self.fitting(self.X, self.residual)
			basic_regressor = BasicRegressor(best_split, c1, c2)
			self.func.append(basic_regressor)
			it += 1

	def fitting(self, X, residual):
		best_split = 0
		best_c1 = 0
		best_c2 = 0
		min_error = 10000
		for spp in self.split_point:
			c1 = 0; c2 = 0
			n1 = 0; n2 = 0
			for (x, r) in zip(X, residual):
				if x < spp:
					c1 += r
					n1 += 1
				else:
					c2 += r
					n2 += 1
			c1 /= float(n1)
			c2 /= float(n2)
			err = sum([(y-c1)**2 if x < spp else (y-c2)**2 for (x, y) in zip(X, residual)])
			if err < min_error:
				min_error = err
				best_split = spp
				best_c1 = c1
				best_c2 = c2
		return best_split, round(best_c1,2), round(best_c2,2)

	def predict(self, test_X):
		pred = np.zeros(len(test_X))
		for tree in self.func:
			pred += tree.predict(test_X)
		return pred

	def show(self):
		print("Final boosting tree:")
		print("\tf(x) = T_1(x) + T_2(x) + T_3(x) + T_4(x) + T_5(x) + T_6(x)")
		print("where,")
		for idx,tree in enumerate(self.func):
			print("\tT_%d(x) = %.2f if x < %.2f else %.2f" %(idx+1, tree.c1, tree.split_point, tree.c2))

if __name__ == '__main__':
	X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
	split_point = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
	bt = BoostingTree(X, y, split_point)
	bt.train()
	bt.show()
	print("Input test X:\n", X)
	print("Boosting tree predicts:\n", bt.predict(X))





