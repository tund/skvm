from __future__ import division

import time
from os.path import join
import numpy as np
from operator import add

from pyspark import SparkConf, SparkContext

MODEL_NAME = "SkVM"
D = 784
K = 10
NUM_TRAIN_PARTITIONS = 10
NUM_TEST_PARTITIONS = 1

DATA_DIR = "sample_data"
DATA_FILE_TRAIN = "mnist_train"
DATA_FILE_TEST = "mnist_test"


def read_point_batch(iterator):
    lines = list(iterator)
    y = np.zeros(len(lines), dtype=np.int)
    X = np.zeros((len(lines), D))
    for i, line in enumerate(lines):
        items = lines[i].split()
        y[i] = np.int(np.float(items[0]))-1
        num_items = len(items) - 1
        for j in xrange(num_items):
            u, v = items[j+1].split(":")
            X[i, np.int(u)-1] = np.float(v)
            
    return [(X, y)]


def scale_data(X, y):
    y_label = sorted(np.unique(y))
    y_count = np.array([len(np.where(y==i)[0]) for i in y_label])
    y_count = (y_count.min()+1e-8) / (y_count+1e-8)
    X *= y_count[y].reshape(len(y), 1)
    return (X, y)

    
def sample_invgau(X, y, w):
    return (X, y, np.random.wald(1 / np.abs(1-(w[y, :]*X).sum(axis=1, keepdims=True)), 1))


def calc_q(X, y, invlb):
    q = np.zeros((K, D))
    for i in xrange(K):
        q[i, :] = ((1+invlb[y==i, :])*X[y==i, :]).sum(axis=0, keepdims=True) - ((1+invlb[y!=i, :])*X[y!=i, :]).sum(axis=0, keepdims=True)
    return q


if __name__ == "__main__":
    conf = SparkConf().setAppName("[{}]".format(MODEL_NAME))
    sc = SparkContext(conf = conf)
    
    start_time = time.time()
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN), NUM_TRAIN_PARTITIONS).mapPartitions(read_point_batch).map(lambda batch: scale_data(batch[0], batch[1]))

    w = 1.0*np.ones((K, D))
    data = data.map(lambda batch: sample_invgau(batch[0], batch[1], w))

    p = data.map(lambda batch: (batch[0]*batch[2]).T.dot(batch[0])).fold(np.eye(D), add)

    q = data.map(lambda batch: calc_q(batch[0], batch[1], batch[2])).reduce(add)

    for i in xrange(K):
        w[i, :] = np.linalg.solve(p, q[i, :])
    print "Training has finished in {:.2f} second(s)".format(time.time() - start_time)
        
    # testing
    data_test = sc.textFile(join(DATA_DIR, DATA_FILE_TEST), NUM_TEST_PARTITIONS).mapPartitions(read_point_batch)
    pred_lab = data_test.map(lambda batch: (batch[0].dot(w.T).argmax(axis=1), batch[1])).cache()
    acc = pred_lab.map(lambda (pred, lab): (pred == lab).sum()).sum() / pred_lab.map(lambda (pred, lab): len(pred)).sum()
    print "Testing accuracy = {:.2f}".format(acc)
    
    sc.stop()

