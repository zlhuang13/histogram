import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from scipy.spatial import cKDTree

class HistogramClassifier():
    def __init__(self):
        self.partition = 2
        self.histogram = {}
        self.kdtrees = None
        self.cells = []
        self.error_count = 0
        self.distance_bound = 1/self.partition
    
    def get_histogram(self):
        return self.histogram

    def cal_index(self, x):
        indices = np.rint(self.partition * x)
        idx = indices.tobytes()
        return idx, indices
                
    def fit(self, X_train, y_train, b, prune = False):
        """
        Parameters
        -------------
        x : list, set of features
        y : 0 or 1, set of labels in order wrt x
        b : positive integer, cell-width parameters
        """
        self.partition = b
        self.distance_bound = len(X_train[0])

        for i in range(len(X_train)):
            idx, indices = self.cal_index(X_train[i])
            try:
                dummy = self.histogram[idx]
            except KeyError:
                self.histogram[idx] = {"indices":indices, "positive":0, "count":0}
            self.histogram[idx]["positive"] += y_train[i]
            self.histogram[idx]["count"] += 1

        self.cells = list(self.histogram.keys())
        self.kdtrees = cKDTree([self.histogram[key]["indices"] for key in self.cells], leafsize = 16)

        for cell_idx in self.histogram.keys():
            self.histogram[cell_idx]["eta"] = self.histogram[cell_idx]["positive"]/self.histogram[cell_idx]["count"]

    def reset(self):
        self.histogram.clear()
        self.partition = 2
        self.kdtrees = None
        self.cells = []
        self.error_count = 0

    def predict(self, x):
        idx, indices = self.cal_index(x)
        try:
            return int(self.histogram[idx]["eta"] > 1/2)
        except KeyError:
            self.error_count += 1
            self.histogram[idx] = {}
            self.histogram[idx]["indices"] = indices
            if self.kdtrees != None:
                neighbour = self.kdtrees.query(indices, k = 1, p = 1, distance_upper_bound = self.distance_bound, workers = -1)[1]
                try:
                    self.histogram[idx]["eta"] = self.histogram[self.cells[neighbour]]["eta"]
                except IndexError:
                    self.histogram[idx]["eta"] = np.random.uniform()
            else:
                self.histogram[idx]["eta"] = np.random.uniform()
            eta = self.histogram[idx]["eta"]
        return int(eta >= 1/2)
