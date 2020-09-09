import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import time


# function to get unique values

def load_data(filepath):
    f = open(filepath)

    # check unique ids
    # x = f.read()
    # x_u = x.replace('\n','').split('\t')
    # len = np.unique(x_u).__len__() #memory err
    # print(len)

    M = []
    for line in f:
        cols = line.replace('\n', '').split('\t')
        M.append(cols)

    return M






# region [Main]
train_x = load_data('./Data/train.txt')  # 20000rows * dynamic
test_x = load_data('./Data/test-public.txt')  # 2001rows incl header * 3


###### NN / MLP Test
#https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron

from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)
res = clf.predict([[2., 2.], [-1., -2.]])
print(res)

#multi-class classification
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)

clf.fit(X, y)
res2 = clf.predict([[1., 2.]])
res3 =clf.predict([[0., 0.]])
print(res2, res3)