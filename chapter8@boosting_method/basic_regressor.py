"""
@author chenjianfeng
@date 2018.01
""" 

class BasicRegressor(object):
	def __init__(self, split_point, c1, c2):
		self.split_point = split_point
		self.c1 = c1
		self.c2 = c2

	def setV(self, split_point):
		self.split_point = split_point
		return self

	def predict(self, X):
		return [self.c1 if x < self.split_point else self.c2 for x in X]