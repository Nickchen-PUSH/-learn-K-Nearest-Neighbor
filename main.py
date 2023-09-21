import numpy as np

import data_loader
from KNN import KNearestNeighbor

knn = KNearestNeighbor()

data_train,lable_train,data_test,lable_test = data_loader.load_CIFAR10("cifar-10-batches-py/")
data_row_train = data_train.reshape(data_train.shape[0],32*32*3)
data_row_test = data_test.reshape(data_test.shape[0],32*32*3)
knn.train(data_row_train,lable_train)

lable_pred = knn.predict(data_row_test)

print("accuracy: %f" % (np.mean(lable_pred==lable_test)))