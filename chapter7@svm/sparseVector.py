"""
@author chenjianfeng
@date 2018.01
""" 

class SparseVector(object):
    def __init__(self, idList, valList):
        if len(idList) != len(valList):
            raise Exception("SparseVector construction error: len(idList)!=len(valList)")
        self.ids = idList
        self.vals = valList
        self.size = len(idList)

    def dot(self, vec):
        p1 = p2 = dot = 0
        while p1 < self.size and p2 < vec.size:
            if self.ids[p1] == vec.ids[p2]:
                dot += self.vals[p1] * vec.vals[p2]
                p1 += 1
                p2 += 1
            elif self.ids[p1] < vec.ids[p2]:
                p1 += 1
            else:
                p2 += 1
        return dot
