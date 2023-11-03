import pandas as pd;
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron

file_path = 'train.csv'
random_state= 42
max_iter= 100

data= pd.read_csv(file_path)
test_data = pd.read_csv('test.csv')

scaler = StandardScaler()

clf = Perceptron(max_iter=max_iter, tol=None, random_state=random_state)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=10)),
    ('perceptron', Perceptron(max_iter=max_iter, tol=None, random_state=random_state))
])

colors= {
    'label_zero': 'blue',
    'label_one': 'red'
}
save_address= 'result/'
plot_size={"x": 10, "y": 6}