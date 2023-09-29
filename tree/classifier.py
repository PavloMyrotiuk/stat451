import numpy as np


def entropy(y):
    total = len(y)
    if total == 0:
        return 0
    result = 0
    bincount = np.bincount(y.astype(np.int64))
    for i in bincount:
        if i != 0:
            result -= i / total * np.log2(i / total)
    return result


class ClassifierNode:
    def __init__(self, samples, depth, max_depth):
        self.samples = samples
        self.max_depth = max_depth
        self.depth = depth

    def _train(self):
        X = self.samples[:, :-1]
        y = self.samples[:, -1]
        self.entropy_ = entropy(y)

        information_gain_value = np.zeros(X.shape[1])
        information_gain_param_value = np.zeros(X.shape[1])

        for col in range(X.shape[1]):
            selected = self.samples[:, [col, -1]]
            selected_sorted = selected[selected[:, 0].argsort()]
            for row in range(selected_sorted.shape[0]):
                value = selected_sorted[row, 0]
                in_gain = self.entropy_ - entropy(selected_sorted[:row, 1]) - entropy(selected_sorted[row:, 1])
                if in_gain > information_gain_value[col]:
                    information_gain_param_value[col] = value
                    information_gain_value[col] = in_gain
        argmax = np.argmax(information_gain_value, axis=None)
        split_condition = self.samples[:, argmax] < information_gain_param_value[argmax]

        self.values_ = np.bincount(y.astype(np.int64))
        self.max_values_class_ = y[np.bincount(y.astype(np.int64)).argmax()]
        stop_children_condition = len(self.values_.nonzero()[0]) == 1 or self.depth == self.max_depth
        self.left_ = None if stop_children_condition else ClassifierNode(self.samples[split_condition, :], self.depth + 1, self.max_depth)._train()
        self.right_ = None if stop_children_condition else ClassifierNode(self.samples[~split_condition, :], self.depth + 1, self.max_depth)._train()
        self.split_function_ = lambda data: data[:, argmax] < information_gain_param_value[argmax]
        return self

    def _predict(self, data):
        if self.left_ is None and self.right_ is None:
            return self.__data_w_current_node_value(data)
        condition = self.split_function_(data)
        if self.left_ is not None:
            subleft = self.left_._predict(data[condition])
        else:
            subleft = self.__data_w_current_node_value(data[condition])
        if self.right_ is not None:
            subright = self.right_._predict(data[~condition])
        else:
            subright = self.__data_w_current_node_value(data[~condition])
        return np.vstack((subleft, subright))

    def __data_w_current_node_value(self, data):
        return np.hstack((data, np.array([self.max_values_class_] * len(data)).reshape(-1, 1)))


class MyrotiukClassifier:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def train(self, data):
        self.root_ = ClassifierNode(samples=data, depth=0, max_depth=self.max_depth)
        self.root_._train()

    def predict(self, data):
        return self.root_._predict(data)
