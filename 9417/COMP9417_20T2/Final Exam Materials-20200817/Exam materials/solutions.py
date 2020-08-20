
## STUDENT ID: z5244467
## STUDENT NAME: Chen Wu

## Question 2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)       # make sure you run this line for consistency 
x = np.random.uniform(1, 2, 100)
y = 1.2 + 2.9 * x + 1.8 * x**2 + np.random.normal(0, 0.9, 100)
plt.scatter(x,y)
plt.show()

## (c)

# YOUR CODE HERE
def change(q,w,e,r):
    d=(1/4)*(w-e-r*q)**2
    f=(d+1)**(1/2)

    return (f-1)
def gdecent(q,w,e,r,d):
    f=e-d*((-w+e+r*q)/((((w-e-r*q)**2)+4)**0.5))
    b=r-d*((q*(-w+e+r*q))/((((w-e-r*q)**2)+4)**0.5))
    return f,b

## plotting help


fig , ax = plt.subplots (3,3, figsize =(10 ,10)) 
alphas = [10e-1, 10e-2, 10e-3,10e-4,10e-5,10e-6,10e-7, 10e-8, 10e-9]


loop,losNum = 100,0
losses=[]
for i in range(0,len(alphas)):
    losses.append([])
    w0,w1 = 1,1
    for j in range(loop):
        l = change(x[j],y[j],w0,w1)
        losNum=losNum+l
        w0,w1=gdecent(x[j],y[j],w0,w1,alphas[i])
        losses[i].append(losNum)
for i, ax in enumerate(ax.flat):
    ax.plot(losses[i])
    ax.set_title(f"step size: {alphas[i]}") # plot titles 
plt.tight_layout () # plot formatting 
plt.show()




## Question 3

# (c)
# YOUR CODE HERE





# Question 5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def create_dataset():
    X, y = make_classification( n_samples=1250,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2,
                                random_state=5,
                                n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size = X.shape)
    linearly_separable = (X, y)
    X = StandardScaler().fit_transform(X)
    return X, y


from pylab import *
X, y=create_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

plt.figure(figsize=(16,8))
subplot(231)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
SVC = SVC()
SVC.fit(X_train, y_train)
plotter(SVC,X, X_test, y_test, "SVC")
subplot(232)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
LR = LogisticRegression()
LR.fit(X_train, y_train)
plotter(LR,X, X_test, y_test, "Logistic Regression")
subplot(233)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
Adaboost = AdaBoostClassifier()
Adaboost.fit(X_train, y_train)
plotter(Adaboost,X, X_test, y_test, "Adaboost")
subplot(234)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
plotter(RF,X, X_test, y_test, "Random Forest")

subplot(235)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
plotter(DT,X, X_test, y_test, "Decision Tree")
subplot(236)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
MLP = MLPClassifier()
MLP.fit(X_train, y_train)
plotter(MLP,X, X_test, y_test, "MLP")


def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                            np.arange(y_min, y_max, plot_step)) 
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)


# (b)
# YOUR CODE HERE

# (c)
# YOUR CODE HERE
