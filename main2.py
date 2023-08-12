# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Data Loading Function
def load_data(path, classes):
    X = []
    Y = []
    for c in classes:
        pth = path + c
        for i in os.listdir(pth):
            img = cv2.imread(pth + '/' + i, 0)
            img = cv2.resize(img, (200,200))
            X.append(img)
            Y.append(classes[c])
    return np.array(X), np.array(Y)

# Data Preprocessing
def preprocess_data(X, Y):
    X_updated = X.reshape(len(X), -1)
    x_train, x_test, y_train, y_test = train_test_split(X_updated, Y, random_state=10, test_size=.20)
    return x_train/255, x_test/255, y_train, y_test  # Feature Scaling

# Model Training
def train_model(x_train, y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)
    return rf_model

# Testing and Visualization
def test_and_visualize(model, test_data_path, classes, best_clf=None):
    y_test = []
    y_pred = []
    dec = {v: k for k, v in classes.items()}  # Reverse dictionary for decoding
    
    # Dictionary for mapping class names to their respective abbreviations
    directory_mapping = {
        'no_tumor': 'NT',
        'pituitary_tumor': 'PT',
        'glioma_tumor': 'GT',
        'meningioma_tumor': 'MT'
    }
    
    for class_name, label in classes.items():
        plt.figure(figsize=(12,8))
        count = 1
        for i in os.listdir(test_data_path + class_name)[:25]:
            plt.subplot(5,5,count)
            img = cv2.imread(test_data_path + class_name + '/' + i, 0)
            img1 = cv2.resize(img, (200, 200))
            img1 = img1.reshape(1, -1)/255
            p = best_clf.predict(img1) if best_clf else model.predict(img1)
            y_pred.append(p[0])
            y_test.append(label)
            plt.title(dec[p[0]])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            # Use the directory_mapping to get the correct abbreviation
            directory_abbr = directory_mapping[class_name]
            plt.savefig(f'predictions/{directory_abbr}/{directory_abbr}_{i}')
            count += 1
    return y_test, y_pred



# Main Execution
path = 'data/Training/'
classes = {'no_tumor': 0, 'pituitary_tumor': 1, 'glioma_tumor': 2, 'meningioma_tumor': 3}
X, Y = load_data(path, classes)
x_train, x_test, y_train, y_test = preprocess_data(X, Y)
model = train_model(x_train, y_train)
y_test, y_pred = test_and_visualize(model, 'data/Testing/', classes)

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy Score: {accuracy}")
