from __future__ import division

import os
import sys
import time

from pyspark import SparkConf, SparkContext
import cPickle as pkl

from os.path import join
import numpy as np


DATA_DIR = "sample_data"
DATA_FILE_TRAIN = "airlines_regression_train"
DATA_FILE_TEST = "airlines_regression_test"

D = 857
NUM_TRAIN_PARTITIONS = 10
NUM_TEST_PARTITIONS = 1
MAX_DISTANCE = 4983.0
EPSILON = np.finfo(float).eps


def read_point_batch(iterator):
    lines = list(iterator)
    y = np.zeros(len(lines))
    X = np.zeros((len(lines), D))
    for i, line in enumerate(lines):
        items = lines[i].split()
        y[i] = np.float(items[0])
        num_items = len(items) - 1
        for j in xrange(num_items):
            u, v = items[j+1].split(":")
            if np.int(u) == D:
                X[i, np.int(u)-1] = np.float(v) / MAX_DISTANCE
            else:
                X[i, np.int(u)-1] = np.float(v)
            
    return [(X, y)]


def sample_invgau(X, y, w):
    return (X, y, np.random.wald(1 / np.abs(y-X.dot(w).ravel()), 1), np.random.wald(1 / np.abs(y-X.dot(w).ravel()), 1))


if __name__ == "__main__":
    conf = SparkConf().setAppName("[SkVM][Airlines] Regression: predict delay minutes")
    sc = SparkContext(conf = conf)

    start_time = time.time()
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN), NUM_TRAIN_PARTITIONS).mapPartitions(read_point_batch)
    
    w = 0.001*np.ones((D, 1))
    
    data = data.map(lambda b: sample_invgau(b[0], b[1], w)).cache()

    p = np.eye(D)
    p += data.map(lambda b: (b[0]*(b[2]+b[3])[:, np.newaxis]).T.dot(b[0])).sum()

    q = data.map(lambda b: (b[2]+b[3]).dot(b[0])).sum()

    w = np.linalg.solve(p, q)

    print "Training has finished in {:.2f} second(s)".format(time.time() - start_time)
    
    # prediction
    X_test = sc.textFile(join(DATA_DIR, DATA_FILE_TEST), NUM_TEST_PARTITIONS).mapPartitions(read_point_batch)
    mae = X_test.map(lambda x: np.sum(np.abs(x[0].dot(w).ravel() - x[1]))).sum() / X_test.map(lambda x: len(x[1])).sum()

    print "MAE = %.2f" % mae
    
    sc.stop()

    