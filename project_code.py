# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:38:52 2018

@author: Parag Kedia
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree

context=pd.read_csv('Dataset.csv')
context.head(5)
context.describe()
print(context.shape)
print(context.groupby('class').size())
context.hist()

context.plot(kind='density', subplots=True, layout=(10,10), sharex=False)

#context.plot(kind='box', subplots=True, layout=(11,11), sharex=False, sharey=False, title='Box Plot for each input variable')

scatter_matrix(context)
plt.show()


correlations = context.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',	'fM3Long',	'fM3Trans',	'fAlpha','fDist',	'class']
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

plt.plot(context['fAlpha'], context['fDist'], color='g')
plt.xlabel('Angle of major axis with vector to origin(degree)')
plt.ylabel('Distance from origin to center of ellipse(mm)')
plt.title('Angle vs Distance')
plt.show()

plt.plot(context['fSize'], context['fDist'], color='g')
plt.xlabel('10-log of sum of content of all pixels')
plt.ylabel('Distance from origin to center of ellipse(mm)')
plt.title('Pixel contents vs Distance')
plt.show()

plt.plot(context['fAlpha'], context['fSize'], color='g')
plt.xlabel('Angle of major axis with vector to origin(degree)')
plt.ylabel('10-log of sum of content of all pixels')
plt.title('Angle vs Pixel contents')
plt.show()

plt.plot(context['fConc'], context['fConc1'], color='g')
plt.xlabel('Ratio of sum of two highest pixels over pixel contents')
plt.ylabel('Ratio of highest pixels over pixel contents')
plt.title('Ratio of sum of two highest pixels vs Ratio of highest pixels')
plt.show()


for cat in ['class']:
    print("Levels for catgeory '{0}': {1}".format(cat, context[cat].unique()))
    
context['class']=context['class'].map({'g':0,'h':1})

X = context.drop(labels='class', axis=1)
X.head()
pd.isnull(X).any()

y = context.iloc[:,-1]
y=y.to_frame(name=None)
y.head()
pd.isnull(y).any()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
from sklearn import tree
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def Gini_index(X_train, X_test, y_train):
    clf_gini = tree.DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    tree.export_graphviz(clf_gini,out_file='tree.dot')
    return clf_gini
     
def Entropy(X_train, X_test, y_train):
    clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(X_train, y_train)
    tree.export_graphviz(clf_entropy,out_file='tree1.dot')
    return clf_entropy
 
def predict_func(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
     
def accuracy(y_test, y_pred):
    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ("Accuracy: ",accuracy_score(y_test,y_pred)*100)
    print("Classification Report: ",classification_report(y_test, y_pred))
 
def main():
    clf_gini = Gini_index(X_train, X_test, y_train)
    clf_entropy = Entropy(X_train, X_test, y_train)
    print("Prediction using Gini Index:")
    y_pred_gini = predict_func(X_test, clf_gini)
    accuracy(y_test, y_pred_gini)
    print("Prediction using Entropy:")
    y_pred_entropy = predict_func(X_test, clf_entropy)
    accuracy(y_test, y_pred_entropy)
     
if __name__=="__main__":
    main()


    
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', degree=2, random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
print("The prediction rate using Linear kernel is:",(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))

classifier_1 = SVC(kernel = 'poly', degree=3, random_state = 0)
classifier_1.fit(X_train, y_train)
y_pred = classifier_1.predict(X_test)
cm_1 = confusion_matrix(y_test, y_pred)
cm_1
print("The prediction rate using Polynomial kernel is:",(cm_1[0,0]+cm_1[1,1])/(cm_1[0,0]+cm_1[0,1]+cm_1[1,0]+cm_1[1,1]))

classifier_2 = SVC(kernel = 'rbf', degree=1, random_state = 0)
classifier_2.fit(X_train, y_train)
y_pred = classifier_2.predict(X_test)
cm_2 = confusion_matrix(y_test, y_pred)
cm_2
print("The prediction rate using rbf kernel is:",(cm_2[0,0]+cm_2[1,1])/(cm_2[0,0]+cm_2[0,1]+cm_2[1,0]+cm_2[1,1]))



from sklearn.linear_model import LogisticRegression
import time
start = time.time()
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred)
cm2
print("The prediction rate is:",(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]))
end = time.time()
print("The time required to compute is:",end - start)

start2 = time.time()
classifier2 = LogisticRegression(random_state = 0,solver='sag')
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
cm2
print("The prediction rate is:",(cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]))
end2 = time.time()
print("The time required to compute is:",end2 - start2)



from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred)
cm3
print("The prediction for RandomForest using entropy is:",(cm3[0,0]+cm3[1,1])/(cm3[0,0]+cm3[0,1]+cm3[1,0]+cm3[1,1]))


classifier3_1 = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier3_1.fit(X_train, y_train)
y_pred = classifier3_1.predict(X_test)
cm3_1 = confusion_matrix(y_test, y_pred)
cm3_1
print("The prediction rate for RandomForest using gini index is:",(cm3_1[0,0]+cm3_1[1,1])/(cm3_1[0,0]+cm3_1[0,1]+cm3_1[1,0]+cm3_1[1,1]))






