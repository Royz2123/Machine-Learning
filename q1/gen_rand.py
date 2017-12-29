import random
import sys

SAMPLES = 300


def main():
	std_dev = 1.0
	mean = 0.0

	if len(sys.argv) >= 2:
		mean = float(sys.argv[1])
	if len(sys.argv) >= 3:
		std_dev = float(sys.argv[2])

	with open("N_%.2f_%.2f" % (mean, std_dev), 'w') as f:
		for i in xrange(SAMPLES):
			f.write("%s\n" % str(random.gauss(mean, std_dev)))


if __name__ == "__main__":
	main()
