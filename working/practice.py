import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed

path = os.listdir('data/Training/')
classes = {'no_tumor': 0, 'pituitary_tumor': 1, 'glioma_tumor': 2, 'meningioma_tumor': 3}

def process_images(c):
    pth = 'data/Training/'+c
    X = []
    Y = []
    for i in os.listdir(pth):
        img = cv2.imread(pth + '/' + i, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[c])
    return X, Y

results = Parallel(n_jobs=-1)(delayed(process_images)(c) for c in classes)

X = []
Y = []
for result in results:
    X.extend(result[0])
    Y.extend(result[1])

X = np.array(X)
Y = np.array(Y)
X_updated = X.reshape(len(X), -1)

x_train, x_test, y_train, y_test = train_test_split(X_updated, Y, random_state=10, test_size=.20)

# FEATURE SCALING
x_train = x_train/255
x_test = x_test/255

rf_model = RandomForestClassifier(n_jobs=-1)
rf_model.fit(x_train, y_train)

# Adjusting the Hyper-parameters
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}



clf = GridSearchCV(rf_model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(x_train, y_train)

# Function for testing
def testing(tumor_type, code):
    dec = {0:'No Tumor', 1:'Pituitary Tumor', 2: 'Glioma Tumor', 3: 'Meningioma Tumor'}
    y_test = []
    y_pred = []
    plt.figure(figsize=(12,8))
    c=1
    for i in os.listdir(f'data/Testing/{tumor_type}')[:25]:
        plt.subplot(5,5,c)
        img = cv2.imread(f'data/Testing/{tumor_type}/'+i, 0)
        img1 = cv2.resize(img, (200, 200))
        img1 = img1.reshape(1, -1)/255
        p = best_clf.predict(img1)
        y_pred.append(p[0])
        y_test.append(code)
        plt.title(dec[p[0]])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'predictions/{tumor_type[:2]}/{tumor_type[:2]}_{i}')
        c+=1
    return y_test, y_pred

# Testing the model with unseen data
y_tests = []
y_preds = []
for tumor_type, code in classes.items():
    y_test, y_pred = testing(tumor_type, code)
    y_tests.extend(y_test)
    y_preds.extend(y_pred)

accuracy = accuracy_score(y_tests, y_preds)
print(f"Overall Accuracy Score: {accuracy}")
