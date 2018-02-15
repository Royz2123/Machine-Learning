import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from sklearn import linear_model, metrics
from sklearn import preprocessing
from scipy import spatial
import scipy

import math
import random
import sys

TRAIN_PATH = "train_data.csv"
TEST_PATH = "test_data.csv"


def read_text(f_name):
    data = []
    with open(f_name, 'r') as f:
        for line in f.readlines():
            data.append(np.array([float(var) for var in line.split(",")]))
    return np.array(data)


# normalize a set of training samples
def normalize(data):
    # normalize to mean 0 and variance 1 (axis 1 is the rows)
    data = preprocessing.scale(data, axis=1)

    # compute the mean vector
    data -= data.mean(axis=0)

    return data


def faces_mean(face1, face2):
    return (face1 + face2) * 0.5


def find_pca(data):
    # compute scatter matrix
    # scatter_matrix = np.cov(data, rowvar=False)
    scatter_matrix = np.dot(data, data.T)

    # compute eigenvalues and eigenvectors
    d, u = scipy.linalg.eigh(scatter_matrix, eigvals=(50, len(data)-1))

    # w_pca projection matrix
    w_pca = data.T.dot(np.asarray(u[::-1])).dot(np.diag(d))

    return w_pca


# Given normalized data find the optimal projection
def find_mda(data):
    proj_data = data.dot(find_pca(data))

    # compute ni, mu_i, mu
    ni = 2
    mu_is = np.array([
        faces_mean(proj_data[i], proj_data[i+1])
        for i in range(0, len(proj_data), 2)
    ])
    mu = np.mean(proj_data, axis=0)

    # calculate s_w
    subs = np.array([
        proj_data[i] - mu_is[int(i/2)]
        for i in range(100)
    ])
    s_w = np.zeros((50,50), dtype = 'float32')
    for i in range(subs.shape[0]):
        s_w = np.add(s_w, np.outer(subs[i].T, subs[i]))

    # calculate s_b
    s_b = np.zeros((50,50), dtype = 'float32')
    for i in range(len(mu_is)):
        s_b = np.add(s_b, ni * np.outer((mu_is[i] - mu), (mu_is[i] - mu).T))

    # calculate w_mda using generalized egien value
    d, u = scipy.linalg.eig(a=s_b, b=s_w)
    return np.asarray(u).T[::-1]


def classify(model, test_data, w_pca, w_opt):
    # first normalize the data
    normalize(test_data)

    # project the test sampels to w_opt
    test_data = test_data.dot(w_pca).dot(w_opt.T)

    # test using nearest neighbour
    tree = spatial.KDTree(model)
    hits = sum([
        index == tree.query(face)[1]
        for index, face in enumerate(test_data)
    ])

    # output the success rate
    return float(hits) / float(test_data.shape[0])

def main():
    # read the data
    train_data = read_text(TRAIN_PATH)
    test_data = read_text(TEST_PATH)

    # normalize data firs
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    # find the optimal projection on training data
    w_pca = find_pca(train_data)
    w_opt = find_mda(train_data)

    # find training model
    model = np.array([
        faces_mean(train_data[i], train_data[i+1])
        for i in range(0, len(train_data), 2)
    ])
    model = model.dot(w_pca).dot(w_opt.T)

    # classify based on testing samples
    success_rate = classify(model, test_data, w_pca, w_opt)
    print ("Success rate:\t%f" % success_rate)

if __name__ == "__main__":
	main()
