import cv2
import sys
import copy
import glob
import json
import pickle

from adaboost import *
from image_utils import *

weak_classifiers_1 = pickle.load(open("./models/weak_classifiers.pkl", 'rb'))
weak_classifiers_2 = pickle.load(open("./models/weak_classifiers_15.pkl", 'rb'))
directory = sys.argv[1]

cascade = [weak_classifiers_2]
files = glob.glob(directory + "*.jpg")

l = []
for file_name in files:
    img = cv2.imread(file_name, 0)
    shape = img.shape
    ii = integral_image(img)

    sizes = np.arange(1, 15, 1.5) * 19.
    faces = []
    for size in sizes:
        scale = size / 19.
        for i in range(0, shape[0] - int(size), int(size)):
            if i + int(size) >= shape[0]:
                continue 
            for j in range(0, shape[1] - int(size), int(size)):
                if j + int(size) >= shape[1]:
                    continue

                flag = 0
                for s_c in cascade:
                    pred = []
                    alpha_sum = sum(w_c["alpha"] for w_c in s_c)
                    for w_c in s_c:
                        f = None
                        f_shape, bxs = w_c["feature_context"]
                        boxes = copy.deepcopy(bxs)
                        try:
                            for b_ in range(len(boxes)):
                                boxes[b_].l, boxes[b_].b = int(boxes[b_].l * scale), int(boxes[b_].b * scale)
                                boxes[b_].x, boxes[b_].y = int(boxes[b_].x * scale), int(boxes[b_].y * scale)

                            if f_shape == [2, 1]:
                                x, y = boxes[0].x + i, boxes[0].y + j
                                f1 = get_integral_sum_box(ii, box(x, y, boxes[0].l, boxes[0].b))
                                f2 = get_integral_sum_box(ii, box(x + boxes[1].l, y, boxes[1].l, boxes[1].b))
                                f = f1 - f2

                            if f_shape == [1, 2]:
                                x, y = boxes[0].x + i, boxes[0].y + j
                                f1 = get_integral_sum_box(ii, box(x, y, boxes[0].l, boxes[0].b))
                                f2 = get_integral_sum_box(ii, box(x, y + boxes[1].b, boxes[1].l, boxes[1].b))
                                f = f2 - f1

                            if f_shape == [3, 1]:
                                x, y = boxes[0].x + i, boxes[0].y + j
                                f1 = get_integral_sum_box(ii, box(x, y, boxes[0].l, boxes[0].b))
                                f2 = get_integral_sum_box(ii, box(x + boxes[1].l, y, boxes[1].l, boxes[1].b))
                                f3 = get_integral_sum_box(ii, box(x + 2 * boxes[2].l, y, boxes[2].l, boxes[2].b))
                                f = f2 - f1 - f3

                            if f_shape == [1, 3]:
                                x, y = boxes[0].x + i, boxes[0].y + j
                                f1 = get_integral_sum_box(ii, box(x, y, boxes[0].l, boxes[0].b))
                                f2 = get_integral_sum_box(ii, box(x, y + boxes[1].b, boxes[1].l, boxes[1].b))
                                f3 = get_integral_sum_box(ii, box(x, y + 2 * boxes[2].b, boxes[2].l, boxes[2].b))
                                f = f2 - f1 - f3

                            if f_shape == [2, 2]:
                                x, y = boxes[0].x + i, boxes[0].y + j
                                f1 = get_integral_sum_box(ii, box(x, y, boxes[0].l, boxes[0].b))
                                f2 = get_integral_sum_box(ii, box(x + boxes[1].l, y, boxes[1].l, boxes[1].b))
                                f3 = get_integral_sum_box(ii, box(x, y + boxes[2].b, boxes[2].l, boxes[2].b))
                                f4 = get_integral_sum_box(ii, box(x + boxes[3].l, y + boxes[3].b, boxes[3].l, boxes[3].b))
                                f = f1 - f2 - f3 + f4
                        except:
                            pass

                        parity, theta = w_c["parity"], w_c["theta"]
                        neg = -0.8
                        if f is None:
                            temp = neg
                        else:
                            temp = 1 if (parity * f) < (parity * theta) else neg
                        pred.append(w_c["alpha"] * temp)

                    if sum(pred) >= alpha_sum * 0.5:
                        size = min(size, shape[0], shape[1])
                        flag += 1 
                if flag == len(cascade):
                    ix, jx = min(i + int(size), shape[0]), min(j + int(size), shape[1])
                    faces.append([(i, j), (ix, jx)])

    for pt in faces:
        img = cv2.rectangle(img, pt[0], pt[1], (0, 255, 0), 3)
        l.append({"iname": file_name.split("/")[-1], "bbox": [pt[0][0], pt[0][1], pt[1][0] - pt[0][0], pt[1][1] - pt[0][1]]})


with open("results.json", 'w') as f:
    f.write(json.dumps(l))
