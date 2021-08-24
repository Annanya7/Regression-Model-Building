# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:10:59 2021

@author: hp
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1)
df=pd.DataFrame({'X':X.flatten(),'Y':y})
print(X)
print(y)
#plot regression dataset
plt.scatter(X,y,marker='.')
plt.show()
#%%
X=df.iloc[:, :-1].values
y=df.iloc[:,1].values

#%%
#Splitting into training and testing
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.10, random_state=1) 


#%%
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

#%%
# Create linear regression object
reg = linear_model.LinearRegression()

#%%
#Train the model
reg.fit(X_train,y_train)


#%%
#Coefficients 
print("Coeficients",reg.coef_)

#%%
#Intercept
print("Intercept",reg.intercept_)

#%%
plt.hist(np.squeeze(X))



