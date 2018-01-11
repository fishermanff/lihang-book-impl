"""
@author chenjianfeng
@date 2018.01
""" 

from __future__ import print_function
import numpy as np

class DecisionTree(object):
    def __init__(self, epsilon = 0.01):
        self.epsilon = epsilon
        self.tree = {}

    def calcEntropy(self, labels):
        if len(labels) == 0:
            raise Exception("Dataset is empty.")
        labelCounts = self.labelCounts(labels)
        entropy = 0
        for k in labelCounts.keys():
            prob = labelCounts[k]/float(len(labels))
            entropy += (-prob * np.log2(prob)) if prob > 0.0 else 0.0
        return entropy

    def calcInfoGain(self, dataset, featureIndex):
        datasetEntropy = self.calcEntropy(dataset[:,-1])
        featureValues = dataset[:,featureIndex]
        valueCounts = {}
        for i in featureValues:
            if valueCounts.has_key(i):
                valueCounts[i] += 1
            else:
                valueCounts[i] = 1
        featureDatasetEntropy = 0
        for k in valueCounts.keys():
            dk = [sample[-1] for sample in dataset if sample[featureIndex] == k]
            featureDatasetEntropy += valueCounts[k]/float(len(featureValues)) * self.calcEntropy(np.array(dk))
        return (datasetEntropy - featureDatasetEntropy)

    def calcInfoGainRatio(self, dataset, featureIndex):
        datasetEntropy = self.calcEntropy(dataset[:,-1])
        featureValues = dataset[:,featureIndex]
        valueCounts = {}
        for i in featureValues:
            if valueCounts.has_key(i):
                valueCounts[i] += 1
            else:
                valueCounts[i] = 1
        featureDatasetEntropy = 0
        featureEntropy = 0
        for k in valueCounts.keys():
            dk = [sample[-1] for sample in dataset if sample[featureIndex] == k]
            prob = valueCounts[k]/float(len(featureValues))
            featureDatasetEntropy += prob * self.calcEntropy(np.array(dk))
            featureEntropy += (-prob * np.log2(prob)) if prob > 0.0 else 0.0
        return (datasetEntropy - featureDatasetEntropy) / featureEntropy

    def findBestFeatureToSplit(self, dataset, featureNames, algorithm='ID3'):
        criteria = {}
        if algorithm == 'ID3':
            for idx, name in enumerate(featureNames):
                criteria[name] = self.calcInfoGain(dataset, idx)
        elif algorithm == 'C4.5':
            for idx, name in enumerate(featureNames):
                criteria[name] = self.calcInfoGainRatio(dataset, idx)
        else:
            raise Exception("Algorithm is not specified. Specifying 'ID3' or 'C4.5'.")   
        maxCriteria = 0
        bestFeature = ''
        for k in criteria.keys():
            if criteria[k] >= maxCriteria:
                maxCriteria = criteria[k]
                bestFeature = k
        return bestFeature, maxCriteria

    def labelCounts(self, labels):
        labelDict = {}
        for i in labels:
            if labelDict.has_key(i):
                labelDict[i] += 1
            else:
                labelDict[i] = 1
        return labelDict

    def majorLabel(self, labels):
        labelCounts = self.labelCounts(labels)
        maxNum = 0
        bestLabel = ''
        for k in labelCounts.keys():
            if labelCounts[k] >= maxNum:
                maxNum = labelCounts[k]
                bestLabel = k
        return bestLabel

    def creatTree(self, dataset, featureNames, algorithm='ID3'):
        labels = dataset[:,-1]
        if list(labels).count(labels[0]) == len(labels):
            return labels[0]
        if len(featureNames) == 0:
            return self.majorLabel(labels)
        bestFeature, maxGain = self.findBestFeatureToSplit(dataset, featureNames, algorithm)
        if maxGain < self.epsilon:
            return self.majorLabel(labels)
        tree = {bestFeature:{}}
        bestFeatureIndex = featureNames.index(bestFeature)
        val2subset = {}
        for sample in dataset:
            if val2subset.has_key(sample[bestFeatureIndex]):
                val2subset[sample[bestFeatureIndex]].append(sample)
            else:
                val2subset[sample[bestFeatureIndex]] = [sample]
        del featureNames[bestFeatureIndex]
        for k in val2subset.keys():
            tree[bestFeature][k] = self.creatTree(np.delete(np.array(val2subset[k]), bestFeatureIndex, axis=1), featureNames)
        return tree

if __name__ == '__main__':
    featureNames = ['age', 'job', 'house', 'credit']
    dataset = np.array([[0, 0, 0, 0, 'no'],
                        [0, 0, 0, 1, 'no'],
                        [0, 1, 0, 1, 'yes'],
                        [0, 1, 1, 0, 'yes'],
                        [0, 0, 0, 0, 'no'],
                        [1, 0, 0, 0, 'no'],
                        [1, 0, 0, 1, 'no'],
                        [1, 1, 1, 1, 'yes'],
                        [1, 0, 1, 2, 'yes'],
                        [1, 0, 1, 2, 'yes'],
                        [2, 0, 1, 2, 'yes'],
                        [2, 0, 1, 1, 'yes'],
                        [2, 1, 0, 1, 'yes'],
                        [2, 1, 0, 2, 'yes'],
                        [2, 0, 0, 0, 'no']
                        ])
    features, labels = dataset[:,0:4], dataset[:,4]
    print("features:\n", features)
    print("labels:\n", labels)
    decisionTree = DecisionTree()
    tree = decisionTree.creatTree(dataset, featureNames, algorithm='ID3')
    print("decisionTree:", tree)



