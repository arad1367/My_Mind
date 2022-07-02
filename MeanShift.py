# based on region of interest and center of mass

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import cycle

# import data
data = pd.read_excel("Taylor2.xlsx")
x = data.drop(columns=["Platform"])
y = data["Platform"]

# Train and Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=10)
print(x_train.shape)
print(x_test.shape)

# model and accuracy
accuracy = []
bandwidth = []
for i in range(1, 12):
    model = MeanShift(bandwidth=i)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # print (model.score(x_test, y_test))
    score = accuracy_score(y_test, y_predict)
    print(score)
    labels = model.labels_
    cluster_centers = model.cluster_centers_
    bandwidth.append(i)
    accuracy.append(score)


# plot of MeanShift
colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
data = data.to_numpy()

plt.subplot(1, 3, 1)
# plt.scatter(data[:, 0], data[:, 1], marker='o', s=18)
plt.scatter(data[:, 1], data[:, 3], marker='o', s=60, c=y)
plt.xlabel("MeanShift Algoritm")
plt.ylabel("Based on region of interest / center of mass")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(data[:, 1], data[:, 3])
plt.xlabel("Before run the algorithm")
plt.legend()


plt.subplot(1, 3, 3)
plt.plot(accuracy, label="Accuracy")
plt.xticks(range(0, len(bandwidth)), bandwidth)
# plt.plot(test_accuracy, label="Test")
# plt.plot(train_accuracy, label="Train")
plt.xlabel("bandwidth")
plt.ylabel("Accuracy score")
plt.legend()
plt.show()
