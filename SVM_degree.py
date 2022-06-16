# Degree of SVM
# Cluster : ML

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Support vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from SVM_cluster import Kernel

# import data
data = pd.read_excel("Start_demo.xlsx")
x = data.drop(columns=["Media"])
y = data["Media"]

# Another import way
# data = data.to_numpy()
# x = data[:, [1, 2]]
# x = data[:, 0:4]
# y = data[:, 4]
print(x)
print("\n", y)

# Visualization
pd.plotting.scatter_matrix(data, c=y, figsize=[10, 10], s=150)
plt.show()

# Create a model and fit

# Train and Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape)
print(x_test.shape)

# Model accuracy and score
test_accuracy = []
train_accuracy = []
Degree = []

for i in range(1, 10):
    model = SVC(kernel="poly", degree=i)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # print (model.score(x_test, y_test))
    score = accuracy_score(y_test, y_predict)

    test_accuracy.append(model.score(x_test, y_test))
    train_accuracy.append(model.score(x_train, y_train))
    Degree.append(i)

    # print(f"test accuracy is : {test_accuracy},  train accuracy is: {train_accuracy}, final accuracy is : {score}")

# plots
plt.plot(Degree, test_accuracy, label="Test")
plt.plot(Degree, train_accuracy, label="Train")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# MSE
mse = mean_squared_error(y_test, y_predict)
print(f"MSE is : {mse}")

# cross validation
# from sklearn.linear_model import LinearRegression
reg = SVC()
cv_score = cross_val_score(reg, x, y, cv=5)
mse = np.mean(cv_score)
print("cross validation mse is: ", mse)

# Over fitting
degree = np.arange(1, 10)
train_accuracy = np.empty(len(degree))
test_accuracy = np.empty(len(degree))

for i, k in enumerate(degree):
    knn_model = SVC(degree=k)
    knn_model.fit(x_train, y_train)
    train_accuracy[i] = knn_model.score(x_train, y_train)
    test_accuracy[i] = knn_model.score(x_test, y_test)

plt.plot(degree, train_accuracy, label="train_accuracy")
plt.plot(degree, test_accuracy, label="test_accuracy")
plt.legend()
plt.xlabel("Number of degree")
plt.ylabel("accuracy")
plt.show()

# Cross validate
# S2
clf = SVC(kernel='poly', degree=9, C=1, random_state=42)
scores = cross_val_score(clf, x, y, cv=5)
print(f"Score is : {scores}")

# S3
print("%0.2f accuracy with a standard deviation of %0.2f" %
      (scores.mean(), scores.std()))

# S4
# build the model
svm_model = SVC(kernel='poly', C=1, degree=9, gamma='auto', random_state=0)

# create the Cross validation object
loo = LeaveOneOut()

# calculate cross validated (leave one out) accuracy score
scores = cross_val_score(svm_model, x, y, cv=loo, scoring='accuracy')

print("Score is : >>>>> ", scores.mean())
