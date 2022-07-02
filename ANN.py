# ANN modelM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# import data
data = pd.read_excel("Hospital.xlsx")
x = data.drop(columns=["Org"])
y = data["Org"]
print(x)
print("\n", y)

# Train and Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=30)
print(x_train.shape)
print(x_test.shape)

# Model accuracy and score
test_accuracy = []
train_accuracy = []
HLS = []   # hidden lyer size

# for i in range [5, 10, 15, 20]
for i in range(30, 40):
    for j in range(35, 45):
        model = MLPClassifier(hidden_layer_sizes=(
            i, j), activation='relu', solver='adam')
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        # print (model.score(x_test, y_test))
        score = accuracy_score(y_test, y_predict)
        print(f"Score is : {score}")

        test_accuracy.append(model.score(x_test, y_test))
        train_accuracy.append(model.score(x_train, y_train))
        HLS.append([i, j])

# plots
plt.plot(test_accuracy, label="Test")
plt.plot(train_accuracy, label="Train")
plt.xticks(range(0, len(HLS)), HLS, rotation=90)
plt.xlabel("Hidden layer size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# cross validation
# from sklearn.linear_model import LinearRegression
reg = MLPClassifier()
cv_score = cross_val_score(reg, x, y, cv=5)
mse = np.mean(cv_score)
print("cross validation mse is: ", mse)

# S2
clf = MLPClassifier(hidden_layer_sizes=(
    i, j), activation='relu', solver='adam')
scores = cross_val_score(clf, x, y, cv=5)
print(f"Score is : {scores}")
