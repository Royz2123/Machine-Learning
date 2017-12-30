import math
import random
import sys

SAMPLES = 300




def read_samples(filename):
    samples = []
    with open(filename, 'r') as f:
        buf = ''
        while True:
            buf = f.readline()
            if not buf:
                break
            samples.append(buf[:-1].split(','))
    return samples

def main():
    # read the samples
    trainA = read_samples("csv_type/train_dataA.csv")
    trainB = read_samples("csv_type/train_dataB.csv")
    testA = read_samples("csv_type/test_dataA.csv")
    testB = read_samples("csv_type/test_dataB.csv")
    validA = read_samples("csv_type/valid_dataA.csv")
    validB = read_samples("csv_type/valid_dataB.csv")

    print(trainA)

if __name__ == "__main__":
	main()
