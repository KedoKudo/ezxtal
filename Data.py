#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-


import numpy as np
import random
import sys


class KmeansCluster(object):
    """pure python implementation of k-means clustering"""
    def __init__(self, data):
        """copy data into separate memory for clustering"""
        self._data = np.array(data)

    def get_cluster(self, num_cluster, max_iter=10, sort_range=None):
        """generate @num_cluster of clusters using @max_iter
        @para:
             num_cluster:   number of clusters
             max_iter   :   maximum number of initials guess allowed,
             sort_range :   the range of the vector elements sorted, ex (0,10)"""
        inertia = sys.float_info.max
        fin_cluster = []
        if sort_range is None:
            sort_range = (0, len(self._data[0]))  # use full vector
        for it in range(max_iter):
            centers = self._init_guess(num_cluster)
            cluster = self._cluster_pts(centers, sort_range)
            new_centers = self._recal_centers(cluster)
            while(not self._isconverged(centers, new_centers)):
                centers = new_centers
                cluster = self._cluster_pts(centers, sort_range)
                new_centers = self._recal_centers(cluster)
            # calculate inertia to evaluate the performance of this cluster
            new_inertia = self._calc_inertia(centers, cluster)
            if (new_inertia < inertia):
                inertia, fin_cluster = new_inertia, cluster
        #finish 10 rounds
        return [inertia, fin_cluster]

    @staticmethod
    def _calc_inertia(center, cluster):
        """calculate inertia/error to evaluate the performance of cluster"""
        return np.sum(np.sum([[np.linalg.norm(np.array(v) - np.array(center[i])) ** 2
                              for v in cluster[i]]
                              for i in range(len(center))]))

    def _init_guess(self, num_cluster):
        """generate initial guess of the centers for each cluster"""
        sample_idx = [random.randint(0, len(self._data)-1) for i in range(num_cluster)]
        return self._data[sample_idx]

    def _cluster_pts(self, centers, sort_range):
        """Lloyd’s algorithm for performing k-means clustering"""
        # reference:
        # https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
        cluster = {}
        for v in self._data:
            full_v = v
            v = v[sort_range[0]: sort_range[1]]  # slice the vector into desired range
            bestcenterkey = min([(i[0], np.linalg.norm(v-centers[i[0]][sort_range[0]: sort_range[1]]))
                                 for i in enumerate(centers)], key=lambda t:t[1])[0]
            try:
                cluster[bestcenterkey].append(full_v)
            except KeyError:
                cluster[bestcenterkey] = [full_v]
        return cluster

    @staticmethod
    def _recal_centers(clusters):
        """reevaluate the center of each cluster"""
        new_centers = []
        keys = sorted(clusters.keys())
        for k in keys:
            new_centers.append(np.mean(clusters[k], axis=0))
        return new_centers

    @staticmethod
    def _isconverged(centers, new_centers):
        return (set([tuple(i) for i in centers]) == set([tuple(j) for j in new_centers]))


if __name__ == "__main__":
    print "Test for k-means clustering"
    import matplotlib.pyplot as plt
    # generate 3 sets of normally distributed points around
    # different means with different variances
    pt1 = np.random.normal(1, 0.2, (100,2))
    pt2 = np.random.normal(2, 0.5, (300,2))
    pt3 = np.random.normal(3, 0.3, (100,2))
    # slightly move sets 2 and 3 (for a prettier output)
    pt2[:,0] += 1
    pt3[:,0] -= 0.5
    xy  = np.concatenate((pt1, pt2, pt3))

    # test cluster engine
    num_cluster = 3
    my_cluster = KmeansCluster(xy)
    err, rst = my_cluster.get_cluster(num_cluster, max_iter=10, sort_range=(0, 2))

    # plot results
    clr = ['red', 'green', 'blue', 'black']
    plt.figure()
    for i in range(num_cluster):
        data = np.array(rst[i])
        x = data[:,0]
        y = data[:,1]
        plt.scatter(x,y,c=clr[i],alpha=0.5)

    plt.show()