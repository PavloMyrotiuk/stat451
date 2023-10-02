from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split

from tree.classifier import MyrotiukClassifier
from util.util import sort_by_first_column

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

feature_labels = np.hstack((X_train, y_train.reshape(-1, 1)))

pamyr_classifier = MyrotiukClassifier(max_depth=2)
pamyr_classifier.train(feature_labels)
pamyr_result = pamyr_classifier.predict(X_test)

sklearn_classifier = DecisionTreeClassifier(criterion='entropy',
                                            max_depth=2,
                                            random_state=1)
sklearn_classifier.fit(X_train, y_train)
sklearn_result = np.hstack((X_test, sklearn_classifier.predict(X_test).reshape(-1, 1)))

print(f"Actual: {sort_by_first_column(pamyr_result)}")
print(f"Expected: {sort_by_first_column(sklearn_result)}")
