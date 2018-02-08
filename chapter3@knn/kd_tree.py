"""
@author chenjianfeng
@date 2018.02
""" 

from __future__ import print_function
import numpy as np

class TreeNode(object):
	def __init__(self, val, split_dim):
		self.val = val
		self.split_dim = split_dim
		self.left = None
		self.right = None

class KdTree(object):
	def __init__(self, data_set):
		self.k = len(data_set[0])
		self.root = self.creatNode(1, data_set)

	def creatNode(self, curr_height, data_set):
		if not data_set:
			return None
		sp_dim = (curr_height-1) % self.k
		data_set.sort(key=lambda x: x[sp_dim])
		median = len(data_set) / 2
		node = TreeNode(data_set[median], sp_dim)
		node.left = self.creatNode(curr_height+1, data_set[:median])
		node.right = self.creatNode(curr_height+1, data_set[median+1:])
		return node

	def preOrderTraversal(self, root):
		if root:
			print(root.val)
			self.preOrderTraversal(root.left)
			self.preOrderTraversal(root.right)

	def searchNearest(self, target):
		stack = []
		curr_nearest_node = None
		curr_nearest_dist = 100000
		return self.search(self.root, target, stack, curr_nearest_node, curr_nearest_dist)

	def search(self, root, target, stack, curr_nearest_node, curr_nearest_dist):
		if not root:
			while len(stack) > 0:
				if not curr_nearest_node:
					curr_nearest_node = stack.pop()
				else:
					node = stack.pop()
					dist = self.distance(node.val, target)
					if dist <= curr_nearest_dist:
						curr_nearest_node = node
						curr_nearest_dist = dist
					if (target[node.split_dim] <= node.val[node.split_dim]) and \
							(target[node.split_dim] + curr_nearest_dist >= node.val[node.split_dim]):
						curr_nearest_node = self.search(node.right, target, stack, curr_nearest_node, curr_nearest_dist)
					elif (target[node.split_dim] >= node.val[node.split_dim]) and \
							(target[node.split_dim] - curr_nearest_dist <= node.val[node.split_dim]):
						curr_nearest_node = self.search(node.left, target, stack, curr_nearest_node, curr_nearest_dist)
			return curr_nearest_node
		stack.append(root)
		if target[root.split_dim] < root.val[root.split_dim]:
			nearest_node = self.search(root.left, target, stack, curr_nearest_node, curr_nearest_dist)
		else:
			nearest_node = self.search(root.right, target, stack, curr_nearest_node, curr_nearest_dist)
		return nearest_node

	def distance(self, a, b):
		d = 0
		for i in range(len(a)):
			d += (a[i] - b[i]) ** 2
		return d ** (1.0/2)

if __name__ == '__main__':
	# data_set = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
	np.random.seed(10)
	data_set = np.random.randint(1,100,(500,2)).tolist()
	kdTree = KdTree(data_set)
	kdTree.preOrderTraversal(kdTree.root)
	nearest_node = kdTree.searchNearest([10,10])
	print("nearest_node:", nearest_node.val)


