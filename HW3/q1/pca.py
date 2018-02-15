import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from sklearn import linear_model, metrics
from sklearn import preprocessing
from scipy import spatial

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
    print(len(data))
    return np.array(data)


# normalize a set of training samples
def normalize(data):
    # normalize to mean 0 and variance 1 (axis 1 is the rows)
    data = preprocessing.scale(data, axis=1)

    # compute the mean vector
    data -= data.mean(axis=0)


def faces_mean(face1, face2):
    return (face1 + face2) * 0.5


# Given normalized data find the optimal projection
def find_opt_projection(data):
    # compute scatter matrix
    # scatter_matrix = np.cov(data, rowvar=False)
    scatter_matrix = np.dot(data.T, data)

    # compute eigenvalues and eigenvectors
    d, u = np.linalg.eigh(scatter_matrix)

    # w_pca projection matrix
    w_pca = np.asarray(u).T[::-1]

    # compute ni, mu_i, mu
    ni = 2
    mu_is = np.array([
        faces_mean(data[i], data[i+1])
        for i in range(0, len(data), 2)
    ])
    mu = mean(data, axis=0)

    # calculate s_w
    subs = [
        data[i] - mu_is[i/2]
        for i in range(100)
    ]
    s_w = sum([
        np.dot(vec, vec.T)
        for vec in subs
    ])

    # calculate s_b
    s_b = sum([
        ni*(class_mean - mu)*(class_mean - mu).T
        for class_mean in mu_is
    ])

    # calculate w_mda
    d, u = np.linalg.eigh(s_b, s_w)
    w_pca = np.asarray(u).T[::-1]

    return w_mda * w_pca.T



def train(train_data, w_opt):
    # first normalize the data
    normalize(train_data)

    # project the training samples to w_opt
    train_data = train_data * w_opt

    # compute model for each person as average of 2 faces:
    modeled_faces = []
    for i in range(0, len(train_data), 2):
        modeled_faces.append(faces_mean(train_data[i], train_data[i+1]))

    # return the projected faces
    return np.array(modeled_faces) * w_opt


def classify(test_data, proj_faces, w_opt):
    # first normalize the data
    normalize(test_data)

    # project the test sampels to w_opt
    test_data = test_data * w_opt

    # test using nearest neighbour
    tree = spatial.KDTree(proj_faces)
    hits = sum([
        index == tree.query(face)[1]
        for index, face in enumerate(test_data)
    ])

    # output the success rate
    return hits / len(test_data)

def main():
    # read the data
    train_data = read_text(TRAIN_PATH)
    test_data = read_text(TEST_PATH)

    # find the optimal projection on training data
    w_opt = find_opt_projection(train_data)

    # train based on training_data
    proj_faces = train(train_data, w_opt)

    # classify based on testing samples
    success_rate = classify(test_data, proj_faces, w_opt)
    print ("Success rate:\t%d" % success_rate)

if __name__ == "__main__":
	main()
