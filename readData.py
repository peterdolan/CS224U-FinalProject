__author__ = 'SEOKHO'

import csv

def read(filename):
    train = []
    test = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(reader, None)
        for row in reader:
            addList = train
            if row[11] == "TEST":
                addList = test
            addList.append((row[1], row[2], row[4]))
    return (train, test)
