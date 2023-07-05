# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:34:41 2019

@author: Nidhisha MAhilong
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('BreastCancerWisconsinDataSet.csv')
dataset = dataset.dropna(axis=1, how='all')
print(dataset.isna().sum())#check for missing values

print(dataset.head())

print(dataset.describe().T)

nRow, nCol = dataset.shape
print(f'There are {nRow} rows and {nCol} columns')

acc = []
##Distribution graphs (histogram/bar graph) of column data
def Barplot(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int((nCol + nGraphPerRow - 1) / nGraphPerRow)
    plt.figure(num = None, figsize = (3 * nGraphPerRow, 5 * nGraphRow), dpi = 120, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xlabel('Benign/Malignant')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]}')# (column {i})
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.savefig("image/count.png")
    plt.show()
    plt.close()

Barplot(dataset, 1, 1)


# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])

X = dataset[ [ col for col in dataset.columns if col != 'diagnosis' ] ]
y = dataset['diagnosis']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state = 0)
lg.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred_lg = lg.predict(X_test)
cm_lg = confusion_matrix(y_test, y_pred_lg)
print(cm_lg)
print("Logistic Regression accuracy \n",accuracy_score(y_test, y_pred_lg))
acc.append(accuracy_score(y_test, y_pred_lg))

# Decision Tree Classification

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_dt = dt.predict(X_test)
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)
print("Decision Tree accuracy \n",accuracy_score(y_test, y_pred_dt))
acc.append(accuracy_score(y_test, y_pred_dt))

# K-Nearest Neighbors (K-NN)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
print("KNN accuracy \n",accuracy_score(y_test, y_pred_knn))
acc.append(accuracy_score(y_test, y_pred_knn))

# Kernel SVM

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
kernel_svc = SVC(kernel = 'rbf', random_state = 0)
kernel_svc.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_kernel = kernel_svc.predict(X_test)
cm_kernel = confusion_matrix(y_test, y_pred_kernel)
print(cm_kernel)
print("Kernel SVM accuracy \n",accuracy_score(y_test, y_pred_kernel))
acc.append(accuracy_score(y_test, y_pred_kernel))

# Naive Bayes

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_nb = nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(cm_nb)
print("Naive Bayes accuracy \n",accuracy_score(y_test, y_pred_nb))
acc.append(accuracy_score(y_test, y_pred_nb))

# Random Forest Classification

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_rf = rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)
print("Random Forest accuracy \n",accuracy_score(y_test, y_pred_rf))
acc.append(accuracy_score(y_test, y_pred_rf))

# Support Vector Machine (SVM)

# Training the SVM model on the Training set
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 0)
svc.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print(cm_svc)
print("SVM accuracy \n",accuracy_score(y_test, y_pred_svc))
acc.append(accuracy_score(y_test, y_pred_svc))
acc = [round(j * 100,2) for j in acc]

fig = plt.figure(figsize = (10, 6))

# creating the accuracy bar plot
models = ["Logistic","Decision Tree","KNN","Kernel SVM","Naive Bayes","Random Forrest","SVM"]
plt.bar(models, acc, color ='green',width = 0.5)

for i, v in enumerate(acc):
    plt.text(i, v + 0.01, str(v)+"%",horizontalalignment="center",verticalalignment="bottom")
plt.xlabel("Models")
plt.ylabel("Accuracy %")
plt.title("Model Accuracy")
plt.savefig("image/accuracy.png")
plt.show()
