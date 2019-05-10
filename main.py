import glob
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from adaboost import *
from image_utils import *
from collections import Counter
from sklearn.metrics import classification_report


np.random.seed(42)
data = pd.read_pickle("data/training.pkl")
np.random.shuffle(data)
data = data[:5000]

features, feature_context = [], None
y = []

print("Computing train features..")
for i in (range(len(data))):
    if feature_context is None:
        _, feature_context = get_features(data[i][0])
    y.append(data[i][1])

print("Finished computing train features..")

x = np.load("/data/ateendra/features.npy")

# x = np.array(features)
y = np.array(y)

np.save("features.npy", x)

print("\n\nData Stats")
print("Train shape", x.shape)
print("Y distribution", Counter(y.tolist()))
print("\n\n")

weak_classifiers = adaboost((x, y), 15)

for w_c in weak_classifiers:
    w_c["feature_context"] = feature_context[w_c["index"]]

print("Storing weak classifiers..")
pickle.dump(weak_classifiers, open("results/weak_classifiers_15.pkl", 'wb'))

test = pd.read_pickle("data/test.pkl")
np.random.shuffle(test)
test = test[:1000]

test_features = []
y_test = []

print("\n Computing test features..")
for i in (range(len(test))):
    f, _ = get_features(test[i][0])
    test_features.append(f)
    y_test.append(test[i][1])

print("\n\nData Stats")
print("Y Test distribution", Counter(y_test))
print("\n\n")

print("\nFinished computing test features..")

x_test = np.array(test_features)
y_test = np.array(y_test)

print("\nTesting out the model..")
alpha_sum = sum(w_c["alpha"] for w_c in weak_classifiers)

y_pred = []
for x in x_test:
    pred = []
    for w_c in weak_classifiers:
        parity, theta = w_c["parity"], w_c["theta"]
        idx = w_c["index"]
        temp = 1 if parity * x[idx] < parity * theta else 0
        pred.append(w_c["alpha"] * temp)
    if sum(pred) >= alpha_sum * .5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Accuracy:", np.mean(y_pred == y_test))
print(classification_report(y_pred, y_test))

