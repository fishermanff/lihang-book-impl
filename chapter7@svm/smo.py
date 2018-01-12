"""
@author chenjianfeng
@date 2018.01
""" 

# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
from sparseVector import SparseVector

class SMO(object):
    def __init__(self, dataset, labels, C=1.0, tolerance=1e-4, maxIter=10000, kernel=None):
        self.dataset = dataset
        self.yArr = labels
        self.C = C
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.kern = kernel
        self.m = len(dataset)
        self.alphaArr = np.zeros(self.m)
        self.b = 0
        self.eCache = np.zeros((self.m,2))       # cache for Ei
        self.K = np.zeros((self.m,self.m))            # 1st dim stands for availability, 2nd dim stands for value

        random.seed(10)
        for i in range(self.m):
            for j in range(self.m):
                self.K[i,j] = self.kern.kernel(dataset[i], dataset[j])

    def takeStep(self, i1, i2):
        if i1 == i2:
            return 0
        L = H = 0
        eps = 1e-5
        if self.yArr[i1] == self.yArr[i2]:
            L = max(0, self.alphaArr[i1] + self.alphaArr[i2] - self.C)
            H = min(self.C, self.alphaArr[i1] + self.alphaArr[i2])
        else:
            L = max(0, self.alphaArr[i2] - self.alphaArr[i1])
            H = min(self.C, self.C + self.alphaArr[i2] - self.alphaArr[i1])
        if L == H:
            return 0
        eta = 2 * self.K[i1,i2] - self.K[i1,i1] + self.K[i2,i2]
        E1 = self.eCache[i1,1] if self.eCache[i1,0] == 1 else self.calcEi(i1)
        E2 = self.eCache[i2,1] if self.eCache[i2,0] == 1 else self.calcEi(i2)
        if eta >= 0:
            return 0
        alpha2new = self.alphaArr[i2] - self.yArr[i2] * (E1 - E2) / eta
        alpha2new = self.clipAlpha(alpha2new, L, H)
        if abs(alpha2new - self.alphaArr[i2]) < eps:
            self.updateEi(i2)
            return 0
        alpha1new = self.alphaArr[i1] + self.yArr[i1] * self.yArr[i2] * (self.alphaArr[i2] - alpha2new)
        
        # update b
        b1 = self.b - (self.eCache[i1,1] + self.yArr[i1] * self.K[i1,i1] * (alpha1new - self.alphaArr[i1]) + \
                            self.yArr[i2] * self.K[i2,i1] * (alpha2new - self.alphaArr[i2]))
        b2 = self.b - (self.eCache[i2,1] + self.yArr[i2] * self.K[i2,i2] * (alpha2new - self.alphaArr[i2]) + \
                            self.yArr[i1] * self.K[i1,i2] * (alpha1new - self.alphaArr[i1]))
        if alpha1new > 0 and alpha1new < self.C:
            b_new = b1
        elif alpha2new >0 and alpha2new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        self.b = b_new

        # update alpha
        self.alphaArr[i1] = alpha1new
        self.alphaArr[i2] = alpha2new

        # update Ei
        self.updateEi(i1)
        self.updateEi(i2)
        return 1

    def examineExample(self, i1):
        E1 = self.calcEi(i1)
        self.eCache[i1] = [1, E1]
        r1 = self.yArr[i1] * E1
        if (r1 < -self.tolerance and self.alphaArr[i1] < self.C) or (r1 > self.tolerance and self.alphaArr[i1] > 0):
            validEi = []
            nonBound = []
            for idx in range(self.m):
                if self.eCache[idx,0] == 1:
                    validEi.append(idx)
                if self.alphaArr[idx] > 0 and self.alphaArr[idx] < self.C:
                    nonBound.append(idx)
            if len(validEi) > 1:
                i2 = self.heuriSelectSecondExample(E1, validEi)
                if self.takeStep(i1,i2):
                    return 1
            # loop over all non-zero and non-C alpha, starting at random point
            ranIndex = random.sample(range(len(nonBound)), len(nonBound))
            for i2 in ranIndex:
                if self.takeStep(i1,i2):
                    return 1
            # loop over all possible i1, starting at random point
            ranIndex = random.sample(range(self.m), self.m)
            for i2 in ranIndex:
                if self.takeStep(i1,i2):
                    return 1
        return 0

    def mainRoutine(self):
        examineAll = True
        it = 0
        numChanged = 0
        while (it < self.maxIter) and ((numChanged > 0) or examineAll):
            if it%100 == 0:
                print("iterating %d" %it)
            numChanged = 0
            if examineAll:
                # loop i1 over all training examples
                for i1 in range(self.m):
                    numChanged += self.examineExample(i1)
            else:
                # loop i1 over examples where alpha is non-0 and non-C
                for i1 in range(self.m):
                    if self.alphaArr[i1] != 0 and self.alphaArr[i1] != self.C:
                        numChanged += self.examineExample(i1)
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
            it += 1

    def heuriSelectSecondExample(self, E1, validEiSet):
        maxDeltaE = 0
        index = 0
        for k in range(len(validEiSet)):
            if abs(self.eCache[k,1] - E1) > maxDeltaE:
                index = k
                maxDeltaE = abs(self.eCache[k,1] - E1)
        return index

    def calcEi(self, i):
        return sum(self.alphaArr * self.yArr * self.K[:,i]) + self.b - self.yArr[i]

    def updateEi(self, i):
        self.eCache[i] = [1, self.calcEi(i)]

    def clipAlpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        return alpha