import numpy as np
import pandas as pd
from sklearn import preprocessing

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('input/sample_fe.csv')
dataset = df.values

X = dataset[:, 3:5]
y = dataset[:, -1]

# in case
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_results():
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X, y, cv=kfold)

model = Sequential()
model.add(Dense(256, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

preds = model.predict(X_test)

for i in range(0, len(preds)):
    if preds[i] >= .5:
        preds[i] = 1
    else:
        preds[i] = 0

print(classification_report(y_test, preds))