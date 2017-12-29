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


def main():
    if len(sys.argv) < 2:
        print("Usage: python est_parms.py [FILENAME]")

    samples = get_samples(sys.argv[1])

    # estimate parms
    mean = est_mean(samples)
    std_dev = est_std_dev(samples, mean)

    # print results
    print("""
    Estimated parameters:
    MEAN:               %f
    STANDARD DEVIATION: %f
    """ % (
        mean, std_dev
    ))




if __name__ == "__main__":
	main()
