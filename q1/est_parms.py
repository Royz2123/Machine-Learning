import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

import math
import random
import sys

SAMPLES = 300

def get_samples(filename):
    samples = []
    with open(filename, 'r') as f:
        buf = ''
        while True:
            buf = f.readline()
            if not buf:
                break
            samples.append(float(buf[:-1]))
    return samples

def est_mean(samples):
    return sum(samples) / len(samples)

def est_std_dev(samples, est_mean):
    return math.sqrt(sum(map(lambda x: (x - est_mean)**2, samples)) / len(samples))

def est_parms(filename):
    samples = get_samples(filename)

    # estimate parms
    mean = est_mean(samples)
    std_dev = est_std_dev(samples, mean)

    # return values
    return mean, std_dev


def main():
    if len(sys.argv) < 3:
        print("Usage: python est_parms.py [FILENAME1] [FILENAME2] ")

    # find the parameters
    mean1, std_dev1 = est_parms(sys.argv[1])
    mean2, std_dev2 = est_parms(sys.argv[2])

    # plot the results
    x = np.linspace(-5, 5, 100)
    plt.plot(x,mlab.normpdf(x, mean1, std_dev1))
    plt.plot(x,mlab.normpdf(x, mean2, std_dev2))
    plt.show()




if __name__ == "__main__":
	main()
