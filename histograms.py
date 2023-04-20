import numpy as np
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

    
class PrivHistogramClassifier():
    def __init__(self, privacy_loss):
        self.partition = 2
        self.histogram = {}
        self.labels = {}
        self.prob = {}
        self.kdtrees = None
        self.cells = []
        self.error_count = 0
        self.distance_bound = 1/self.partition
        self.privacy_loss = privacy_loss
    
    def get_histogram(self):
        return self.histogram

    def cal_index(self, x):
        indices = np.rint(self.partition * x)
        idx = indices.tobytes()
        return idx, indices

    def cal_index_set(self, x):
        indices = np.rint(self.partition * x)
        idx = [i.tobytes() for i in indices]
        return idx, indices
                
    def fit(self, X_train, y_train, b):
        """
        Parameters
        -------------
        x : list, set of features
        y : 0 or 1, set of labels in order wrt x
        b : positive integer, cell-width parameters
        """
        self.partition = b
        self.distance_bound = len(X_train[0])
        self.dimension = len(X_train[0])
        idx, indices = self.cal_index_set(X_train)
        for i in range(len(X_train)):
            try:
                dummy = self.histogram[idx[i]]
            except KeyError:
                self.histogram[idx[i]] = indices[i]
                self.labels[idx[i]] = []
            self.labels[idx[i]].append(y_train[i])

        self.cells = list(self.histogram.keys())
        self.kdtrees = cKDTree(list(self.histogram.values()))

        for cell_idx in self.cells:
            ones = sum(self.labels[cell_idx])
            size = len(self.labels[cell_idx])
            eta = ones/size
            noise = np.random.laplace(0,1/(size*self.privacy_loss))
            eta = eta + noise
            eta = min(1, max(eta, 0))
            self.prob[cell_idx] = eta


    def reset(self):
        self.histogram.clear()
        self.labels.clear()
        self.prob.clear()
        self.partition = 2
        self.kdtrees = None
        self.cells = []
        self.error_count = 0

    def predict(self, x):
        idx, indices = self.cal_index(x)
        try:
            eta = self.prob[idx]
        except KeyError:
            self.error_count += 1
            self.histogram[idx] = indices
            if self.kdtrees != None:
                neighbour = self.kdtrees.query(indices, k = 1, p = 1, distance_upper_bound = self.distance_bound, workers = -1)[1]
                self.prob[idx] = self.prob[self.cells[neighbour]]
            else:
                self.prob[idx] = np.random.uniform()
            eta = self.prob[idx]
        return int(eta >= 1/2)

if __name__ == "__main__":
    hist = HistogramClassifier()
    X_train = [[0,0],[0,0.1],[0,0.2],[0.5,0.7],[0.7,0.5],[0.5,0.6]]
    y_train = [1,1,1,0,0,0]
    hist.fit(X_train, y_train)
    test = [0.3,0.3]
    label = hist.predict(test)
    print(label)
