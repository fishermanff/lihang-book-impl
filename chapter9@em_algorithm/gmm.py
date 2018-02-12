"""
@author chenjianfeng
@date 2018.02
""" 

from __future__ import print_function
import numpy as np

class GaussianMixtureModel(object):
	def __init__(self, model_num, model_parameters):
		self.model_num = model_num
		self.truth_alpha = model_parameters[0]
		self.truth_mu = model_parameters[1]
		self.truth_sigma2 = model_parameters[2]
		self.y = self.generateData(self.truth_alpha, self.truth_mu, self.truth_sigma2, 500)
		self.estimated_alpha = []
		self.estimated_mu = []
		self.estimated_sigma2 = []

	def generateData(self, truth_alpha, truth_mu, truth_sigma2, data_num):
		y = np.zeros(data_num)
		for j in range(data_num):
			r = np.random.rand()
			if r < truth_alpha[0]:
				y[j] = np.random.normal(truth_mu[0], np.sqrt(truth_sigma2[0]))
			elif r < truth_alpha[0] + truth_alpha[1]:
				y[j] = np.random.normal(truth_mu[1], np.sqrt(truth_sigma2[1]))
			else:
				y[j] = np.random.normal(truth_mu[2], np.sqrt(truth_sigma2[2]))
		return y

	def forward(self, y, mu, sigma):
		return 1.0/(np.sqrt(2*np.pi)*sigma) * np.exp(-((y-mu)**2)/(2*sigma**2))

	def emEstimating(self, inits_parameters=[], max_iter=100):
		curr_alpha = inits_parameters[0]
		curr_mu = inits_parameters[1]
		curr_sigma2 = inits_parameters[2]
		N = len(self.y)
		K = self.model_num
		it = 0
		while it < max_iter:
			gamma = np.zeros([N,K])

			# E-step:
			for k in range(K):
				gamma[:,k] = curr_alpha[k] * self.forward(self.y, curr_mu[k], np.sqrt(curr_sigma2[k]))
			for j in range(N):
				gamma[j,:] = gamma[j,:] / float(sum(gamma[j,:]))

			# M-step:
			for k in range(K):
				nk = sum(gamma[:,k])
				curr_alpha[k] = sum(gamma[:,k]) / float(N)
				curr_sigma2[k] = sum(gamma[:,k] * (self.y-curr_mu[k])**2) / float(nk)
				curr_mu[k] = sum(gamma[:,k] * self.y) / float(nk)

			it += 1

		self.estimated_alpha = curr_alpha
		self.estimated_mu = curr_mu
		self.estimated_sigma2 = curr_sigma2

if __name__ == '__main__':
	np.random.seed(47)
	model_num = 3
	#[[alpha1, ..., ...], [mu1, ..., ...], [sigma21, ..., ...]]
	model_parameters = [[0.2, 0.3, 0.5], [10, 20, 30], [1, 1.5, 2]]
	gmm = GaussianMixtureModel(model_num, model_parameters)
	print("observations:\n", gmm.y)
	gmm.emEstimating(inits_parameters=[[0.33, 0.33, 0.34], [15, 25, 35], [2, 2, 2]], max_iter=1000)
	print("truth_alpha: %s >>> estimated_alpha: %s" %(gmm.truth_alpha, gmm.estimated_alpha))
	print("truth_mu: %s >>> estimated_mu: %s" %(gmm.truth_mu, gmm.estimated_mu))
	print("truth_sigma2: %s >>> estimated_sigma2: %s" %(gmm.truth_sigma2, gmm.estimated_sigma2))



