__author__ = 'SEOKHO'

import readData
from bagsvm import BagSvm
from w2vPool import W2VPool

def main():
    train, test = readData.read("SICK.txt")
    clf = W2VPool()
    clf.train(train)
    clf.test(test)


if __name__ == "__main__":
    main()