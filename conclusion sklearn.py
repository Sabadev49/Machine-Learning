#project1

#knn(K-Nearest Neighbors)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
dataset=load_iris()
X=dataset.data
y=dataset.target
scaler=StandardScaler()
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
trainscaler=scaler.fit_transform(X_train)
testscaler=scaler.transform(X_test)
for k in [1,3,5,7,9]:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(trainscaler,y_train)
    score=cross_val_score(model,X_train, y_train,cv=3)
    final_score=model.score(testscaler,y_test)
    print(f"K={k}, Score={final_score}")
print("-"*30)
#project 2
#comparison with or without scaler
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset=load_wine()
X=dataset.data
y=dataset.target
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model_scale=KNeighborsClassifier(n_neighbors=5)
model_noscale=KNeighborsClassifier(n_neighbors=5)
scaler=StandardScaler()
train_scaler=scaler.fit_transform(X_train)
test_scaler=scaler.transform(X_test)
model_scale.fit(train_scaler,y_train)
model_noscale.fit(X_train,y_train)
accuracy_with_scale=model_scale.score(test_scaler,y_test)
accuracy_without_scale=model_noscale.score(X_test,y_test)
print("model accuracy with scale:",accuracy_with_scale)
print("model accuracy without scale:",accuracy_without_scale)
print("-"*30)
#project 3
# This program uses 5-fold cross-validation to test different K values in KNN
# and finds the K that gives the highest average validation accuracy.
# The data is standardized before training to make the features comparable.
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
dataset=load_iris()
X=dataset.data
y=dataset.target
scaler=StandardScaler()
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
train_scale=scaler.fit_transform(X_train)
test_scale=scaler.transform(X_test)
best_k=0
best_score=0
for k in range(1,15,2):
    model=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(model,train_scale,y_train,cv=5)
    print(score.mean())
    if score.mean()>best_score:
        best_score=score.mean()
        best_k=k
print("best score:",best_score)
print("best k:",best_k)
print("-"*30)
#project 4
# This program uses 5-fold cross-validation to find the best K value for KNN.
# The data is standardized before training, and the best K is then used
# to train the final model and evaluate its accuracy on the test set.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
dataset=load_iris()
X=dataset.data
y=dataset.target
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42,test_size=0.2)
scaler=StandardScaler()
train_scaler=scaler.fit_transform(X_train)
test_scaler=scaler.transform(X_test)
best_k=0
best_score=0
for k in range(1,15,2):
    model=KNeighborsClassifier(k)
    score=cross_val_score(model,X_train,y_train,cv=5)
    if best_score < score.mean():
        best_k=k
        best_score=score.mean()
final_model=KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(train_scaler,y_train)
final_accuracy=final_model.score(test_scaler,y_test)
print(final_accuracy)
print("best k:",best_k)
print("best score:",best_score)
print("-"*30)
#       Quiz decision Tree, random forest, svm
#project 1(Decision Tree)
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
dataset=load_breast_cancer()
X, y=dataset.data, dataset.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for m in range(1,6):
    model=DecisionTreeClassifier(max_depth=m,random_state=42)
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    print(f"deps={m}\naccuracy:{score:.2f}")
print("-"*30)
#     project2, Random Forest and Decision Tree 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
dataset=load_wine()
X,y=dataset.data,dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
start=time.time()
tree_model=DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train,y_train)
tree_score=tree_model.score(X_test,y_test)
end=time.time()
starts=time.time()
forest_model=RandomForestClassifier()
forest_model.fit(X_train,y_train)
forest_score=forest_model.score(X_test,y_test)
ends=time.time()
print(f"Tree accuracy:{tree_score:.2f}, Tree time:{end-start:.4f}s")
print(f"forest accuracy:{forest_score:.2f}, Forest time:{ends-starts:.4f}s")
print("-"*30)
#project 3, SVM, Goal: find the better kernel for load_iris datasets
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
datasets=load_iris()
X,y=datasets.data, datasets.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_linear=SVC(kernel='linear',random_state=42)
model_linear.fit(X_train,y_train)
model_rbf=SVC(random_state=42)
model_rbf.fit(X_train,y_train)
model_poly=SVC(kernel='poly',degree=3,random_state=42)
model_poly.fit(X_train,y_train)
linear_accuracy=model_linear.score(X_test,y_test)
rbf_accuracy=model_rbf.score(X_test,y_test)
poly_accuracy=model_poly.score(X_test,y_test)
print(f"kernel:linear,accuracy:{linear_accuracy:.2f}")
print(f"kernel:rbf,accuracy:{rbf_accuracy:.2f}")
print(f"kernel:poly,accuracy:{poly_accuracy:.2f}") 
print("-"*30)
#project4       Comparison XGB & Random forest 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
data=load_breast_cancer()
X,y=data.data, data.target
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42,test_size=0.2)
xgmodel=XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
xgmodel.fit(X_train,y_train)
fomodel=RandomForestClassifier(n_estimators=100,random_state=42)
fomodel.fit(X_train,y_train)
xgbscore=xgmodel.score(X_test,y_test)
forestscore=fomodel.score(X_test,y_test)
print(f"xgboost score:{xgbscore:.2f}\nrandom forest score:{forestscore:.2f}")
print("-"*30)
#project5      impact Learning rate
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
data=load_breast_cancer()
X,y=data.data, data.target
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42,test_size=0.2)
for lr in np.arange(0.1,0.5,0.05):
    model=XGBClassifier(n_estimators=100,learning_rate=lr,max_depth=3,random_state=42)
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    print(f"learning rate:{lr:.2f}\nscore:{score:.2f}")
# Supervised Learning finished

