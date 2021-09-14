# -*- coding: utf-8 -*-
"""
Created on Tue Sep 7 15:53:07 2021

@author: hp
"""


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import metrics
from sklearn.model_selection import train_test_split

#%%
X, Y = make_regression(n_samples= 100 , noise=5 , random_state= 5 , n_features=5 , n_targets= 1)
#%%
print(X.shape , Y.shape)
#%%
df = pd.DataFrame(
{'Feature_1':X[:,0],
'Feature_2':X[:,1],
'Feature_3':X[:,2],
'Feature_4':X[:,3],
'Feature_5':X[:,4],
'Target':Y
}
)
df.head()
#%%
print(df.describe())

#%%
plt.scatter(df["Feature_1"] , df["Target"] , color = "red")
plt.xlabel("Feature 1")
plt.ylabel("Target")

#%%
plt.scatter(df["Feature_2"] , df["Target"] , color = "red")
plt.xlabel("Feature 2")
plt.ylabel("Target")
#%%
plt.scatter(df["Feature_3"] , df["Target"] , color = "red")
plt.xlabel("Feature 3")
plt.ylabel("Target")
#%%
plt.scatter(df["Feature_4"] , df["Target"] , color = "red")
plt.xlabel("Feature 4")
plt.ylabel("Target")
#%%
plt.scatter(df["Feature_5"] , df["Target"] , color = "red")
plt.xlabel("Feature 5")
plt.ylabel("Target")
#%%
n = int(input("Enter the degree : "))
poly = PolynomialFeatures(n)
#%%
X_poly = poly.fit_transform(X)
#%%
X_poly.shape
#%%
sc = StandardScaler()
X_standard = sc.fit_transform(X_poly)
#%%
Y_arr = np.reshape(Y ,(100 ,1))
Y_standard = sc.fit_transform(Y_arr)

#%%
X_train ,X_test ,Y_train , Y_test = train_test_split(X_standard , Y_standard , test_size = 0.3 , shuffle = True )
#%%
model = RidgeCV(normalize = True , cv = 5 )
model.fit(X_train , Y_train)
Y_pred = model.predict(X_test)
model.score(X_test , Y_pred)

#%%

print("Mean Squared Error is",metrics.mean_squared_error(Y_test,Y_pred))
print("Coefficient of determination regression score function:-\n",metrics.r2_score(Y_test, Y_pred))