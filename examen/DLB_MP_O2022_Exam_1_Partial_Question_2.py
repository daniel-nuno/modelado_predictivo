# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:06:11 2022
@author: Act. Daniel Lagunas Barba
Exam 1° Partial
"""

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Modeling Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,r2_score)
from sklearn.preprocessing import MinMaxScaler

fig_folder = 'D:/Maestria - Ciencia de Datos/3er Semestre/MP - MODELADO PREDICTIVO (O2022_MCD3396A)/Examenes/1° Parcial/Práctico/'

#%% Importing data
Train = pd.read_csv(fig_folder+'Audit_train.csv', index_col=0)
Test = pd.read_csv(fig_folder+'Audit_test.csv', index_col=0)
Unknown = pd.read_csv(fig_folder+'Audit_unknown.csv', index_col=0)

#%% Transforming and splitting data
Train = Train.drop(['Detection_Risk'], axis=1)
X_unknown = Unknown.drop(['Detection_Risk'], axis=1)
Test = Test.drop(['Detection_Risk'], axis=1)

y_train = Train['Audit_Risk']
X_train = Train.drop('Audit_Risk', axis=1)

y_test = Test['Audit_Risk']
X_test = Test.drop('Audit_Risk', axis=1)

# StandardScaler
sc_X = MinMaxScaler()
sc_X.fit(X_train)

sc_Y = MinMaxScaler()
sc_Y.fit(np.array(y_train).reshape(-1,1))

X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
X_unknown = sc_X.transform(X_unknown)

y_train = sc_Y.transform(np.array(y_train).reshape(-1,1))
y_test = sc_Y.transform(np.array(y_test).reshape(-1,1))

#%% III.- Model Regression
# 1.- Linear Model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

ref = np.linspace(min(y_test),max(y_test))

y_predict = linreg.predict(X_test)
fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predict)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'),plt.ylabel('y predict')
plt.title('Linear regression (original), RMSE=%0.4f, R^2=%0.4f'%(mean_squared_error(y_test,y_predict),r2_score(y_test,y_predict)))
plt.grid()

RMSE_Original = mean_squared_error(y_test,y_predict)
R2_Original = r2_score(y_test,y_predict)
print('Training Set: RMSE=%0.4f, R^2=%0.4f'%(mean_squared_error(y_train,linreg.predict(X_train)), r2_score(y_train,linreg.predict(X_train))))
print('Test Set: RMSE=%0.4f, R^2=%0.4f'%(RMSE_Original, R2_Original))

#%%
Unknown['Audit_Risk'] = sc_Y.inverse_transform(linreg.predict(X_unknown))
Unknown['Audit_Risk'] = np.where(Unknown['Audit_Risk']<0,0,Unknown['Audit_Risk'])
Unknown.to_csv(fig_folder+'Unknown dataset.csv')