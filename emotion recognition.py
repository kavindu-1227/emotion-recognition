# -*- coding: utf-8 -*-
"""


@author: Kavindu Hapuarachchi
"""

#===========================================
#Import Images to Python
#===========================================
#required libraries
import cv2
import os
import numpy as np
import pandas as pd
from skimage import io

#Method to import image folder
images = []
def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images = []

load_images_from_folder("E:/4th yr Project/images/set2/angry/include")
n_angry=len(images) #assign length of list as number of images
angry=pd.DataFrame()

for i in range(n_angry):
    arr=images[i]
    matrix=arr[:,:,2]
    row = matrix.flatten(order='C')
    new=pd.DataFrame(row)
    trans=new.T
    angry=angry.append(trans,ignore_index=True)
    
#=======================================================

images = []
load_images_from_folder("E:/4th yr Project/images/set2/happy/include")
n_happy=len(images) #assign length of list as number of images
happy=pd.DataFrame()

for i in range(n_happy):
    arr=images[i]
    matrix=arr[:,:,2]
    row = matrix.flatten(order='C')
    new=pd.DataFrame(row)
    trans=new.T
    happy=happy.append(trans,ignore_index=True)
#############################################
images = []

load_images_from_folder("E:/4th yr Project/images/set2/sad/include")
n_sad=len(images) #assign length of list as number of images
sad=pd.DataFrame()
for i in range(n_sad):
        arr=images[i]
        matrix=arr[:,:,2]
        row = matrix.flatten(order='C')
        new=pd.DataFrame(row)
        trans=new.T
        sad=sad.append(trans,ignore_index=True)
#################################################


images = []
load_images_from_folder("E:/4th yr Project/images/set2/surprise/include")
n_surprise=len(images) #assign length of list as number of images
surprise=pd.DataFrame()
for i in range(n_surprise):
    arr=images[i]
    matrix=arr[:,:,2]
    row = matrix.flatten(order='C')
    new=pd.DataFrame(row)
    trans=new.T
    surprise=surprise.append(trans,ignore_index=True)
    
########################################################


images = []
load_images_from_folder("E:/4th yr Project/images/set2/neut/include")
n_neut=len(images) #assign length of list as number of images
neut=pd.DataFrame()


for i in range(n_neut):
    arr=images[i]
    matrix=arr[:,:,2]
    row = matrix.flatten(order='C')
    new=pd.DataFrame(row)
    trans=new.T
    neut=neut.append(trans,ignore_index=True)
##############################################


#Create response vector considering the number of images
#for each emotion. 0,3,4,5 represent angry,happy,sad and
#surprise emotion respectively.
#These numbers are repeated according to the number of images
#for each emotion.


y=np.repeat([0,3,4,5],[n_angry,n_happy,n_sad,n_surprise])
#################################


#Create predictor variables
X=pd.DataFrame()
X=X.append(angry,ignore_index=True)
X=X.append(happy,ignore_index=True)
X=X.append(sad,ignore_index=True)
X=X.append(surprise,ignore_index=True)
X=X.append(neut,ignore_index=True)



######################Standerdize dataset
from sklearn import preprocessing
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
X = scaler.fit_transform(X)


#split test set and training set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
##########################


#Libraries
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
# #####################Feature Extraction using PCA


# Compute a PCA (eigenfaces)
n_components = 0.8
#0.8 represents 80% of total variation of the original data
print("Extracting the top %d eigenfaces from %d faces"
% (n_components, X_train.shape[0]))
t0 = time()


pca = PCA(n_components=n_components, svd_solver='full',
whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))


#Project images to eignspace
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
# #############################


# Train a SVM classification model rbf kernel
print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced')
, param_grid)
t0 = time()
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



# ##############################
# Predicting test set using classifier
print("Predicting emotions on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
cm=confusion_matrix(Y_test, y_pred)



#############################
# SVM 3 degree polynomial
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
}
clf = GridSearchCV(SVC(kernel='poly', class_weight='balanced')
, param_grid)
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###########################
# SVM linear kernel
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
}
clf = GridSearchCV(SVC(kernel='linear'
, class_weight='balanced',max_iter=1000000), param_grid)
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



##########################
# SVM sigmoid kernel
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
}
clf = GridSearchCV(SVC(kernel='sigmoid', class_weight='balanced')
, param_grid)
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

###############################
#Classification Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_pca, Y_train)
y_pred = clf.predict(X_test_pca)
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


############################################
#Random Forest
from sklearn.ensemble import RandomForestClassifier
t0 = time()
clf = RandomForestClassifier(n_estimators=1000)
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Predicting emotions on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


###########################################
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
t0 = time()
clf = GaussianNB()
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Predicting emotions on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
##############################################################