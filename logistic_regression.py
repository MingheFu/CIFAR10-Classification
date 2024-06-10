import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to load preprocessed CIFAR-10 data
def load_preprocessed_data():
    data = np.load('data.npy')
    labels = np.load('labels.npy')
    return data, labels

# Function to preprocess data
def preprocess_data(data):
    data = data.astype('float32') / 255.0  # Normalize
    data = data.reshape(data.shape[0], -1)  # Flatten
    scaler = StandardScaler()  # Standardize features
    data = scaler.fit_transform(data)
    return data

# Load CIFAR-10 data
data, labels = load_preprocessed_data() 

# SMOTE and PCA
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Balance data using SMOTE
sm = SMOTE(random_state=1234)
data, labels = sm.fit_resample(data, labels)

# Dimensionality reduction using PCA
pca = PCA(n_components=0.95)
data = pca.fit_transform(data)

# Function to train and evaluate the model
def train_and_evaluate(C, max_iter, tol, solver, penalty, multi_class, fit_intercept, class_weight, test_size):
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=test_size, random_state=1234)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)
    
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    
    model = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver=solver, penalty=penalty, 
                               multi_class=multi_class, fit_intercept=fit_intercept, class_weight=class_weight, verbose=verbose)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f'Test Size: {test_size}')
    print(f'Training Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

# Hyperparameters
C = 0.001
max_iter = 10000
tol = 1e-4
solver = 'lbfgs'
penalty = 'l2'
multi_class = 'multinomial'
fit_intercept = True
class_weight = 'balanced'
verbose = 1

# Test size
test_size = 0.2

print(f'Training model with C={C}, solver={solver}, max_iter={max_iter}, tol={tol}, penalty={penalty}, multi_class={multi_class},fit_intercept={fit_intercept}, class_weight={class_weight}, verbose={verbose}...')
train_and_evaluate(C, max_iter, tol, solver, penalty, multi_class, fit_intercept, class_weight, test_size)
