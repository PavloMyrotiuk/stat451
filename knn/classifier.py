import numpy as np

from priority.min_heap import MyrotiukMinHeap, HeapType


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class CloseNode(HeapType):

    def __init__(self, base_value, sample_value, class_value):
        self.base_value = base_value
        self.sample_value = sample_value
        self.ed = euclidean_distance(base_value, sample_value)
        self.class_value = class_value

    def __gt__(self, other):
        return self.ed > other.ed


class MyrotiukKNNClassifier:
    def __init__(self, samples, k):
        self.samples = samples
        self.k = k
        self.min_heap = MyrotiukMinHeap()

    def classify(self, features):
        shape = features.shape
        result = np.empty((shape[0], shape[1] + 1))
        for f in range(len(features)):
            feature = features[f]
            for sample in self.samples:
                self.min_heap.insert(CloseNode(feature, sample[:-1], sample[-1]))
            f_res = []
            for r in range(self.k):
                f_res.append(self.min_heap.pop().class_value)
            max_class_value = np.argmax(np.bincount(f_res))
            result[f] = np.append(feature, [max_class_value])
            self.min_heap = MyrotiukMinHeap()
        return result
