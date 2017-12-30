import math
import random
import sys

import pnn

def read_samples(filename):
    samples = []
    with open(filename, 'r') as f:
        buf = ''
        while True:
            buf = f.readline()
            if not buf:
                break
            samples.append([int(v) for v in buf[:-1].split(',')])
    return samples

def validate(myPNN):
    validation_samples = {
        "A" : read_samples("csv_type/valid_dataA.csv"),
        "B" : read_samples("csv_type/valid_dataB.csv")
    }

    for sig in range(1, 4, 0.1):
        myPNN._chosen_sig = sig
        suc = myPNN.classify_bunch(validation_samples)
        print(suc)


def main():
    # read the samples
    trainA = read_samples("csv_type/train_dataA.csv")
    trainB = read_samples("csv_type/train_dataB.csv")
    myPNN = pnn.PNN([trainA, trainB])

    # validate(myPNN)

    test_samples = {
        "A" : read_samples("csv_type/test_dataA.csv"),
        "B" : read_samples("csv_type/test_dataB.csv")
    }
    suc = myPNN.classify_bunch(test_samples)
    print("Classification rate:\t%f" % suc)


if __name__ == "__main__":
	main()
