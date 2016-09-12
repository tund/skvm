from __future__ import division

import time
from os.path import join
import numpy as np
from operator import add

from pyspark import SparkConf, SparkContext
import cPickle as pkl
import zipfile

MODEL_NAME = "SkVM"
D = 784
NUM_TRAIN_PARTITIONS = 1
NUM_TEST_PARTITIONS = 1

DATA_DIR = "sample_data"
DATA_FILE_TEST = "mnist_test"


def read_point_batch(iterator):
    lines = list(iterator)
    y = np.zeros(len(lines), dtype=np.int)
    X = np.zeros((len(lines), D))
    for i, line in enumerate(lines):
        items = lines[i].split()
        y[i] = np.int(np.float(items[0]))
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
    return (X, y, np.random.wald(1 / (1e-8+np.abs(1-(w[y, :]*X).sum(axis=1, keepdims=True))), 1))


def calc_q(X, y, invlb, K):
    q0 = np.zeros((K, D))
    q1 = np.zeros((K, D))
    for i in xrange(K):
        q0[i, :] = ((1+invlb[y==i, :])*X[y==i, :]).sum(axis=0, keepdims=True)
        q1[i, :] = ((1+invlb[y!=i, :])*X[y!=i, :]).sum(axis=0, keepdims=True)
    return (q0, q1)


if __name__ == "__main__":
    conf = SparkConf().setAppName("[{}] label-drift classification".format(MODEL_NAME))
    sc = SparkContext(conf = conf)
    
    np.random.seed(6789)
    
    w0 = np.zeros((0, D))
    p = np.eye(D)
    q_diff = np.zeros((1, D))
    q = np.zeros((0, D))
    y_set = set()
    y_list = np.array([], dtype=np.int)
    y_map = {}
    K = 0
    
    acc_list, rtime_list = np.zeros(10), np.zeros(10)

    for b in xrange(10):
        start_time = time.time()
        
        data = sc.textFile(join(DATA_DIR, "mnist_labeldrift/block{}".format(b)), NUM_TRAIN_PARTITIONS*(b+1)).mapPartitions(read_point_batch)
        data = data.filter(lambda x: len(x[1])>0).cache()

        y_unique = set(np.unique(data.flatMap(lambda batch: np.unique(batch[1])).collect()))    
        y_new = y_unique - y_set
        y_set = y_set.union(y_new)
        for i in y_new:
            y_map[i] = K
            y_list = np.append(y_list, i)
            w0 = np.vstack((w0, np.ones((1, D))))
            q = np.vstack((q, -q_diff))
            K += 1

#         print y_set
#         print y_list
#         print y_map
#         print q.shape

        data = data.map(lambda batch: (batch[0], np.array([y_map[i] for i in batch[1]], dtype=np.int)))

        data = data.map(lambda batch: sample_invgau(batch[0], batch[1], w0))

        p += data.map(lambda batch: (batch[0]*batch[2]).T.dot(batch[0])).reduce(add)

        (q0, q1) = data.map(lambda batch: calc_q(batch[0], batch[1], batch[2], K)).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
        q_diff += q0.sum(axis=0, keepdims=True)
        q += (q0 - q1)

        w = np.zeros((K, D))
        for i in xrange(K):
            w[i, :] = np.linalg.solve(p, q[i, :])
            
        rtime_list[b] = time.time() - start_time

        # predict
        data_test = sc.textFile(join(DATA_DIR, DATA_FILE_TEST), NUM_TEST_PARTITIONS).mapPartitions(read_point_batch)

        # Make prediction and test accuracy
        pred_lab = data_test.map(lambda batch: (y_list[batch[0].dot(w.T).argmax(axis=1)], batch[1])).cache()
        acc = pred_lab.map(lambda (pred, lab): (pred == lab).sum()).sum() / pred_lab.map(lambda (pred, lab): len(pred)).sum()
        print "Block {}: Accuracy = {}, Time = {}".format(b, acc, rtime_list[b])
        
        acc_list[b] = acc

    print "Time list = {}".format(rtime_list.cumsum())
    
    sc.stop()
    