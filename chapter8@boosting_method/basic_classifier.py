"""
@author chenjianfeng
@date 2018.01
""" 

class BasicClassifier(object):
	def __init__(self, v, symbol_set=['<', '>']):
		self.v = v
		self.symbol_set = symbol_set
		self.symbol = self.symbol_set[0]

	def setV(self, v):
		self.v = v
		return self

	def setSymbol(self, symbol):
		self.symbol = symbol
		return self

	def predict(self, X):
		if self.symbol == '<':
			return [1 if x < self.v else -1 for x in X]
		else:
			return [1 if x > self.v else -1 for x in X]