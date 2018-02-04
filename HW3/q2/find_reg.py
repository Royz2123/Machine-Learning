import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from sklearn import linear_model, metrics

import math
import random
import sys

DATA_PATH = "regdata.csv"

TRAINING_SIZES = range(10, 100, 10)

def read_text(f_name):
    data = []
    with open(f_name, 'r') as f:
        for line in f.readlines():
            data.append([float(var) for var in line.split(",")])
    return data


def find_square_error(data_x, data_y, reg):
    predicted_y = reg.predict(data_x)
    return sum([
        (real_y - predict_y)**2 for real_y, predict_y
        in zip(data_y, predicted_y)
    ]) / len(data_y)

def predict_model(train_set, test_set):
    reg = linear_model.LinearRegression()
    reg.fit(train_set[0], train_set[1])

    # find the mean square error
    train_error = find_square_error(train_set[0], train_set[1], reg)
    test_error = find_square_error(test_set[0], test_set[1], reg)

    # return errors
    return train_error, test_error


def main():
    # read the data
    data = read_text(DATA_PATH)

    # permute thee order of the samples
    random.shuffle(data)

    # split into data X and output Y
    data_y = [vec[0] for vec in data]
    data_x = [vec[1:] for vec in data]

    # split into training and testing
    train_errors = []
    test_errors = []
    for train_size in TRAINING_SIZES:
        train_error, test_error = predict_model(
            (data_x[:train_size], data_y[:train_size]),
            (data_x[train_size:], data_y[train_size:]),
        )
        # document errors
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.plot(TRAINING_SIZES, train_errors)
    plt.plot(TRAINING_SIZES, test_errors)
    plt.show()



if __name__ == "__main__":
	main()
