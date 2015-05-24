__author__ = 'SEOKHO'

import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVR
import sklearn
import scipy
import numpy as np

class BagSvm:
    def __init__(self):
        self.clf = LinearSVR()
        self.vectorizer = HashingVectorizer()
    def getFeatures(self, data):
        sentenceAs = [data[0] for data in data]
        sentenceBs = [data[1] for data in data]
        scores = [float(data[2]) for data in data]
        sentenceAFeatures = self.vectorizer.fit_transform(sentenceAs)
        sentenceBFeatures = self.vectorizer.fit_transform(sentenceBs)
        features = scipy.sparse.hstack([sentenceAFeatures, sentenceBFeatures])
        return features, scores
    def train(self, trainData):
        features, scores = self.getFeatures(trainData)
        self.clf.fit(features, scores)
    def test(self, test):
        features, scores = self.getFeatures(test)
        results = self.clf.predict(features)
        print(sklearn.metrics.mean_squared_error(results, np.array(scores)))

