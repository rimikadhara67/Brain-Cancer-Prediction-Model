import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2

path = os.listdir('data/Training/')
classes = {'no_tumor': 0, 'pituitary_tumor': 1, 'glioma_tumor': 2, 'meningioma_tumor': 3}
X = []
Y = []
for c in classes:
    pth = 'data/Training/'+c
    for i in os.listdir(pth):
        img = cv2.imread(pth + '/' + i, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[c])

X = np.array(X)
Y = np.array(Y)
X_updated = X.reshape(len(X), -1)

x_train, x_test, y_train, y_test = train_test_split(X_updated, Y, random_state=10, test_size=.20)

# FEATURE SCALING
x_train = x_train/255
x_test = x_test/255

# FEATURE SELECTION - SKIPPED TO AVOID DATA LOSS

# TRAINING THE MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# lg_model = LogisticRegression()
# lg_model.fit(x_train, y_train)

# svc_model = SVC()
# svc_model.fit(x_train, y_train)

# knn_model = KNeighborsClassifier()
# knn_model.fit(x_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# PERFORMANCE ANALYSIS
# print("Logistical Regression model:")
# print(f"----- Training score: {lg_model.score(x_train, y_train)}")
# print(f"----- Testing score: {lg_model.score(x_test, y_test)}")

# print("SVC model:")
# print(f"----- Training score: {svc_model.score(x_train, y_train)}")
# print(f"----- Testing score: {svc_model.score(x_test, y_test)}")

# print("K-Nearest Neighbor model:")
# print(f"----- Training score: {knn_model.score(x_train, y_train)}")
# print(f"----- Testing score: {knn_model.score(x_test, y_test)}")

# print("Random Forest Classifier model:")
# print(f"----- Training score: {rf_model.score(x_train, y_train)}")
# print(f"----- Testing score: {rf_model.score(x_test, y_test)}")


# TESTING THE MODEL WITH UNSEEN DATA
# NOTE: previously we had split the seen data into test and training datasets
# however, now we are applying the model on a compleltely new data set which 
# has not previously been seen by our model
import time
start = time.time()
dec = {0:'No Tumor', 1:'Pituitary Tumor', 2: 'Glioma Tumor', 3: 'Meningioma Tumor'}
y_test = []
y_pred = []

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/no_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/no_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = rf_model.predict(img1)
    y_pred.append(p[0])
    y_test.append(0)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/NT/NT_{i}')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/pituitary_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/pituitary_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = rf_model.predict(img1)
    y_pred.append(p[0])
    y_test.append(1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/PT/PT_{i}')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/glioma_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/glioma_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = rf_model.predict(img1)
    y_pred.append(p[0])
    y_test.append(2)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/GT/GT_{i}')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/meningioma_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/meningioma_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = rf_model.predict(img1)
    y_pred.append(p[0])
    y_test.append(3)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/MT/MT_{i}')
    c+=1

end = time.time()
time = end - start
accuracy = accuracy_score(y_test, y_pred)
print(f"Time: {time}")
print(f"Overall Accuracy Score: {accuracy}")
print(rf_model.get_params())

# Adjusting the Hyper-paramteters
param_grid = [
    {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000]
    }
]

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(rf_model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(x_train, y_train)
print(best_clf.best_estimator_)
end1 = time.time()
time = end1 - end
print(f"Time: {time}")

# Checking accuracy with change in hyper parameters
start = time.time()
dec = {0:'No Tumor', 1:'Pituitary Tumor', 2: 'Glioma Tumor', 3: 'Meningioma Tumor'}
y_test = []
y_pred = []

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/no_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/no_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = best_clf.predict(img1)
    y_pred.append(p[0])
    y_test.append(0)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/NT/NT_{i}')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/pituitary_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/pituitary_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = best_clf.predict(img1)
    y_pred.append(p[0])
    y_test.append(1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/PT/PT_{i}')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/glioma_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/glioma_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = best_clf.predict(img1)
    y_pred.append(p[0])
    y_test.append(2)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/GT/GT_{i}')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir('data/Testing/')
c=1
for i in os.listdir('data/Testing/meningioma_tumor')[:25]:
    plt.subplot(5,5,c)
    
    img = cv2.imread('data/Testing/meningioma_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = best_clf.predict(img1)
    y_pred.append(p[0])
    y_test.append(3)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(f'predictions/MT/MT_{i}')
    c+=1

end = time.time()
time = end - start
accuracy = accuracy_score(y_test, y_pred)
print(f"Time: {time}")
print(f"Testing score: {rf_model.score(y_test, y_pred)}")
print(f"Overall Accuracy Score: {accuracy}")
