__author__ = 'SEOKHO'

from gensim.models import Word2Vec
import readData
from nltk import word_tokenize
import numpy as np
from scipy.spatial.distance import cosine
import sklearn

def train():
    train, test = readData.read("SICK.txt")
    sentences = []
    for trainData in train:
        sentences.append(word_tokenize(trainData[0]))
        sentences.append(word_tokenize(trainData[1]))
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=8)
    for i in range(10):
        model.train(sentences)
    model.save("vectors.bin")

def testPooling():
    mat = np.array([[0.5, 0.6], [0.3, 0.4]])
    print(dynamicPooling(mat, 8))



def simMatrix(model, sentence1, sentence2):
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        mat = np.zeros((len(tokens1), len(tokens2)))
        for index1, token1 in enumerate(tokens1):
            for index2, token2 in enumerate(tokens2):
                vec1 = model[token1] if token1 in model else np.zeros((len(model['the'])))
                vec2 = model[token2] if token2 in model else np.zeros((len(model['the'])))
                mat[index1][index2] = cosine(vec1, vec2)
        return mat

def dynamicPooling(matrix, finalDim):
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
        return finalMatrix

if __name__ == "__main__":
    #train()
    #test()
    testPooling()

