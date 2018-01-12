"""
@author chenjianfeng
@date 2018.01
""" 

class Kernel(object):
	def __init__(self, kernelTup):
		self.kernelTup = kernelTup

	def kernel(self, x1, x2):
		if self.kernelTup[0] == 'linear':
		    return self.linearKernel(x1,x2)
		elif self.kernelTup[0] == 'rbf':
		    return self.rbfKernel(x1,x2)
		else:
		    raise Exception("Please specify a kernel ('linear' or 'rbf').")

	def linearKernel(self, x1, x2):
		return x1.dot(x2)

	def rbfKernel(self, x1, x2):
		sigma = 1.0
		if len(self.kernelTup)==2 and isinstance(self.kernelTup[1],(int, float)):
		    sigma = self.kernelTup[1]
		numerator = SparseVector.dot(x1,x1) - 2*SparseVector.dot(x1,x2) + SparseVector.dot(x2,x2)      # ||x1-x2||^2
		return np.exp(-numerator / (2*sigma**2))