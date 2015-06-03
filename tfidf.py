__author__ = 'JAMES'

import readData
import numpy as np
import string
import math
from collections import Counter


class TFIDF:
    def __init__(self):
        self.train, self.test = readData.read("SICK.txt")
        # self.train = [("A group of kids is playing in a yard and an old man is standing in the background", "A group of boys in a yard is playing and a man is standing in the background")]
        self.vocab = []
        self.dfs = Counter()
        self.tfs = Counter()
        for example in self.train:
            sentence1 = example[0].translate(None, string.punctuation).lower().split(" ")
            sentence2 = example[1].translate(None, string.punctuation).lower().split(" ")
            sentences = [sentence1, sentence2]
            thistfSet = Counter()
            for i in xrange(len(sentences)):
                thistfSet += Counter(sentences[i])
            self.tfs += thistfSet
            self.dfs += Counter(thistfSet.keys())

        self.vocab = self.dfs.keys()

        self.tfidf = {}
        for term in self.tfs.keys():
            self.tfidf[term] = np.log(self.tfs[term]) * np.log(float(len(self.train)) / (self.dfs[term] + 1))


    def __getitem__(self, term):
        if term not in self.vocab:
            return 0.0
        return self.tfidf[term]


def main():
    tfidf = TFIDF()
    print(tfidf['the'])
    print(tfidf['man'])
    print(tfidf['red'])
    print(tfidf['cloak'])
    print(tfidf['dress'])
    print(tfidf.tfs['cloak'])
    print(tfidf.tfs['red'])
    import cPickle
    with open('tfidf.bin', "wb") as f:
        cPickle.dump(tfidf.tfidf, f)


if __name__ == "__main__":
    main()