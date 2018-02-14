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
    return np.array(data)


# normalize a set of training samples
def normalize(data):
    # compute the mean vector
    data -= data.mean(axis=0)

    # normalize to mean 0 and variance 1 (axis 1 is the rows)
    # data = preprocessing.scale(data, axis=1)




# Given normalized data find the optimal projection
def find_opt_projection(data):
    # compute scatter matrix
    scatter_matrix = np.cov(data, rowvar=False)

    # compute eigenvalues and eigenvectors
    d, u = np.linalg.eigh(scatter_matrix)

    # w_pca projection matrix
    w_pca = np.asarray(u).T[::-1]



def train(train_data, w_opt):
    # first normalize the data
    normalize(train_data)

    # project the training samples to w_opt
    train_data = train_data * w_opt

    # compute model for each person as average of 2 faces:
    modeled_faces = []
    for i in range(0, len(train_data), 2)
        modeled_faces.append((train_data[i] + train_data[i+1])*0.5)

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
    w_opt = find_opt_projection(data)

    # train based on training_data
    proj_faces = train(train_data, w_opt)

    # classify based on testing samples
    success_rate = classify(test_data, proj_faces, w_opt)
    print ("Success rate:\t%d" % success_rate)

if __name__ == "__main__":
	main()
