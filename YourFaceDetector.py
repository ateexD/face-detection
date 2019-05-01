import glob
import pickle
import pandas as pd

from adaboost import *
from sklearn.metrics import classification_report
from image_utils import *

np.random.seed(42)
data = pd.read_pickle("data/training.pkl")
data = data[:200]

features, feature_context = [], []
y = []

for i in range(len(data)):
    f, c = get_features(data[i][0])
    features.append(f)
    feature_context.append(c)
    y.append(data[i][1])


x = np.array(features)
y = np.array(y)

weak_classifiers = adaboost((x, y), 5)

pickle.dump(weak_classifiers, open("results/weak_classifiers.pkl", 'wb'))
pickle.dump(feature_context, open("results/train_feature_context.pkl", 'wb'))

test = pd.read_pickle("data/test.pkl")
test = test[:25]

test_features = []
y_test = []

for i in range(len(test)):
    f, _ = get_features(test[i][0])
    test_features.append(f)
    y_test.append(test[i][1])

x_test = np.array(test_features)
y_test = np.array(y_test)


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

