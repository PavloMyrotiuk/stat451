from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

from knn.classifier import MyrotiukKNNClassifier
from sklearn.neighbors import KNeighborsClassifier
from util.util import sort_by_first_column

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

samples = np.hstack((X_train, y_train.reshape(-1, 1)))

classifier = MyrotiukKNNClassifier(samples, 5)
pamyr_result = classifier.classify(X_test)
print(f"Actual {sort_by_first_column(pamyr_result)}")


sklearn_model = KNeighborsClassifier(n_neighbors=5)
sklearn_model.fit(X_train, y_train)
sklearn_result = sklearn_model.predict(X_test)
features_w_classes = np.hstack((X_test, sklearn_result.reshape(-1, 1)))
print(f"Expected {sort_by_first_column(features_w_classes)}")
