import numpy as np


class KNearestNeighbor:

    def __init__(self) -> None:
        pass

    def train(self,x,y):
        self.Xtr = x
        self.Ytr = y
        
    def predict(self,X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test,dtype=self.Ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr-X[i,:]),axis=1)
            min_index = np.argmin(distances)
            Ypred = self.Ytr[min_index]

        return Ypred