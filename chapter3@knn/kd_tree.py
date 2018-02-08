"""
@author chenjianfeng
@date 2018.02
""" 

from __future__ import print_function

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

if __name__ == '__main__':
	data_set = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
	kdTree = KdTree(data_set)
	kdTree.preOrderTraversal(kdTree.root)


