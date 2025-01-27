import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
# import pandas as pd

iris = datasets.load_iris()
classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))

print('Accuracy: %f' % score)
