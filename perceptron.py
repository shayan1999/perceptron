import pandas as pd;
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from variables import *
from plots import *

# question 1
train_data, val_data = train_test_split(data, test_size=50, random_state=random_state)

# question 2
train_data_scaled = scaler.fit_transform(train_data[['x1', 'x2']])
val_data_scaled = scaler.transform(val_data[['x1', 'x2']])

# question 3
actual_data_plot(train_data_scaled, train_data['labels'], 'normalized_data', 'Scatter Plot of Classes')

# question 4
clf.fit(train_data_scaled, train_data['labels'])

train_predictions = clf.predict(train_data_scaled)
first_train_accuracy = accuracy_score(train_data['labels'], train_predictions)

val_predictions = clf.predict(val_data_scaled)
first_val_accuracy = accuracy_score(val_data['labels'], val_predictions)

# decision boundary
x_min, x_max = train_data_scaled[:, 0].min() - 1, train_data_scaled[:, 0].max() + 1
y_min, y_max = train_data_scaled[:, 1].min() - 1, train_data_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

decision_boundary_plot(xx, yy, Z, train_data_scaled, train_data['labels'])

def evaluate_polynomial(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(train_data_scaled)
    X_val_poly = poly.transform(val_data_scaled)
    
    clf.fit(X_train_poly, train_data['labels'])
    
    train_predictions = clf.predict(X_train_poly)
    train_accuracy = accuracy_score(train_data['labels'], train_predictions)
    
    val_predictions = clf.predict(X_val_poly)
    val_accuracy = accuracy_score(val_data['labels'], val_predictions)
    
    wrong_idx = np.where(val_data['labels'] != val_predictions)[0]
    
    return train_accuracy, val_accuracy, wrong_idx

degrees = [2, 3, 5, 10]
results = {}
for degree in degrees:
    train_accuracy, val_accuracy, wrong_idx = evaluate_polynomial(degree)
    results[degree] = {'Train': train_accuracy, 'Validation': val_accuracy, 'Wrongs': wrong_idx}
    plot_wrong_predictions_poly(wrong_idx, train_data_scaled, train_data['labels'], val_data_scaled, val_data['labels'], degree)


test_data_scaled = scaler.transform(test_data[['x1', 'x2']])

pipeline.fit(train_data[['x1', 'x2']], train_data['labels'])
test_predictions = pipeline.predict(test_data[['x1', 'x2']])

result_maker(first_train_accuracy, first_val_accuracy, results, test_predictions)
actual_data_plot(test_data_scaled, test_predictions, 'test_data', 'Test Data Predictions (Normalized)')
