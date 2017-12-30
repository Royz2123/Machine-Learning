import math
import random
import numpy as np
import time

DEFAULT_SIG = 1

class PNN(object):
    CLASSIFCATION = ["A", "B"]

    def __init__(self, training_samples):
        self._training_data = training_samples
        self._chosen_sig = DEFAULT_SIG

    def calc_parzen_window(self, val):
        const = 1.0 / self._chosen_sig * math.sqrt(2 * math.pi)
        return const * math.exp(0.5 * (-val**2 / self._chosen_sig ** 2))

    def calc_y(self, samples, val):
        total = 0
        for sample in samples:
            x = np.linalg.norm(np.array(sample) - np.array(val))
            total += self.calc_parzen_window(x)
        return total / float(len(samples))

    def classify(self, val):
        gaussians = []
        for class_samples in self._training_data:
            gaussians.append(self.calc_y(class_samples, val))
        return PNN.CLASSIFCATION[gaussians.index(max(gaussians))]

    def classify_bunch(self, labeled_samples):
        correct = 0
        overall = 0

        for label, samples in labeled_samples.items():
            for sample in samples:
                # print "%s %s" % (self.classify(sample), label)
                correct += (self.classify(sample) == label)
                overall += 1

        return correct / float(overall)
