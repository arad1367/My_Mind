# GMM >>>> Gaussian Mixture Model >>>> Unsupervised ML
# The programm is useful to cluster and market segmentation

# Libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import data
data = pd.read_excel("Data.xlsx")
x = data.drop(columns=["ind_var"])
y = data["dep_var"]
print(x)
print("\n", y)

# Pairplot
sb.pairplot(data, hue="experience", palette="hls")
plt.show()

# Create a model and fit
# Train and Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=30)
print(x_train.shape)
print(x_test.shape)

# Model accuracy and score
test_accuracy = []
train_accuracy = []
covariance_type = []

for i in ["full", "tied", "diag", "spherical"]:
    model = GaussianMixture(covariance_type=i, n_components=4, random_state=30)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # print (model.score(x_test, y_test))
    score = accuracy_score(y_test, y_predict)

    test_accuracy.append(model.score(x_test, y_test))
    train_accuracy.append(model.score(x_train, y_train))
    covariance_type.append(i)
    # print(f"test accuracy is : {test_accuracy},  train accuracy is: {train_accuracy}, final accuracy is : {score}")

# plots
data = data.to_numpy()
plt.subplot(1, 3, 1)
plt.plot(test_accuracy, label="Test")
plt.plot(train_accuracy, label="Train")
plt.xticks(range(0, len(covariance_type)), covariance_type)
# plt.xticks([0, 1, 2, 3], covariance_type)
plt.xlabel("Covariance_type")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 3, 2)
# chon bakhsh cluster nist va mixture has model.label_ nadarim
labels = model.predict(x)
means = model.means_
# plt.subplot(1, 2, 1)
plt.scatter(data[:, 1], data[:, 2], s=30, c=labels)
# plt.scatter(x[:, 0], x[:, 1], s=10, c=labels)
plt.scatter(means[:, 1], means[:, 2], s=100, c="b", marker="*")
plt.xlabel("Clusters")
plt.ylabel("After GMM ALgorithm")
plt.legend()


plt.subplot(1, 3, 3)
plt.scatter(data[:, 1], data[:, 2])
# plt.scatter(x[:, 0], x[:, 1], s=10, c=labels)
plt.xlabel("Clusters")
plt.ylabel("Before GMM algorithm")
plt.legend()
plt.show()
