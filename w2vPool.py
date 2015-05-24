__author__ = 'SEOKHO'
from sklearn.svm import SVR
from gensim.models import Word2Vec
from nltk import word_tokenize
import numpy as np
from scipy.spatial.distance import cosine
import sklearn

class W2VPool:
    def __init__(self, poolingDim = 20):
        self.clf = SVR(C = 0.5)
        self.model = Word2Vec.load("vectors.bin")
        self.poolingDim = poolingDim
    def getFeatures(self, data):
        sentenceAs = [data[0] for data in data]
        sentenceBs = [data[1] for data in data]
        scores = [float(data[2]) for data in data]
        features = []
        for i in range(len(sentenceAs)):
            mat = self.simMatrix(self.model, sentenceAs[i], sentenceBs[i])
            mat = self.dynamicPooling(mat, self.poolingDim)
            features.append(np.ndarray.flatten(mat))
        return features, scores
    def simMatrix(self, model, sentence1, sentence2):
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        mat = np.zeros((len(tokens1), len(tokens2)))
        for index1, token1 in enumerate(tokens1):
            for index2, token2 in enumerate(tokens2):
                vec1 = model[token1] if token1 in model else np.zeros((len(model['the'])))
                vec2 = model[token2] if token2 in model else np.zeros((len(model['the'])))
                mat[index1][index2] = cosine(vec1, vec2)
        return mat
    def dynamicPooling(self, matrix, finalDim):
        finalMatrix = np.zeros((finalDim, finalDim))
        for i in range(finalDim):
            for j in range(finalDim):
                compressionArea = []
                for a in range(int(float(i) / finalDim * matrix.shape[0]), int(float(i + 1) / finalDim * matrix.shape[0])):
                    for b in range(int(float(j) / finalDim * matrix.shape[1]), int(float(j + 1) / finalDim * matrix.shape[1])):
                        compressionArea.append(matrix[a][b])
                if len(compressionArea) == 0:
                    finalMatrix[i][j] = matrix[int(float(i) / finalDim * matrix.shape[0])][int(float(j) / finalDim * matrix.shape[1])]
                else:
                    finalMatrix[i][j] = min(compressionArea)

        return np.nan_to_num(finalMatrix)

    def train(self, trainData):
        features, scores = self.getFeatures(trainData)
        self.clf.fit(features, scores)
        results = self.clf.predict(features)
        print("Training Error")
        print(sklearn.metrics.mean_squared_error(results, np.array(scores)))

    def test(self, test):
        features, scores = self.getFeatures(test)
        results = self.clf.predict(features)
        print("Testing Error")
        print(sklearn.metrics.mean_squared_error(results, np.array(scores)))
