import glob
import pandas as pd

from adaboost import *
from sklearn.metrics import classification_report
from image_utils import *

np.random.seed(0)
data = (pd.read_pickle("~/Downloads/training.pkl"))
np.random.shuffle(data)

features = [get_features(data[i][0])[0] for i in range(200)]
x = np.array(features)
y = np.array([data[i][1] for i in range(200)])

weak_classifiers = adaboost((x, y), 3)

y_pred = []

xtest = [get_features(data[i][0])[0] for i in range(200, 225)]
xtest = np.array(xtest)
y_test = np.array([data[i][1] for i in range(200, 225)])

alpha_sum = sum(w_c["alpha"] for w_c in weak_classifiers)

for x in xtest:
    pred = []
    for w_c in weak_classifiers:
        parity, theta = w_c["parity"], w_c["theta"]
        idx = w_c["index"]
        temp = 1 if parity * x[idx] < parity * theta else 0
        pred.append(w_c["alpha"] * temp)
    if (sum(pred) >= alpha_sum * .5):
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Accuracy:", np.mean(y_pred == y_test))
print(classification_report(y_pred, y_test))

