# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 21:15:43 2022

@author: Pedro Mtz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,r2_score)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import stats

train = pd.read_csv('../Dataset entrada/Examen 1/Audit_train.csv')
test = pd.read_csv('../Dataset entrada/Examen 1/Audit_test.csv')
data = pd.concat([train, test], ignore_index=True)



# Quitamos variables innecesarias
data = data.drop(["Unnamed: 0", "Detection_Risk"], axis=1)
# Eliminamos outliers
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]



# Se vuelve a barajear el dataset porque tenia valores muy grandes como train
# y muy pequeños como test
X_train, X_test, y_train, y_test = train_test_split(data.drop("Audit_Risk",
                                                              axis=1), 
                                                    data["Audit_Risk"],
                                                    test_size=0.2,
                                                    random_state=40)


# Realizamos modelo
linreg = LinearRegression()
linreg.fit(X_train, y_train)

ref = np.linspace(min(y_test),max(y_test))
y_predict = linreg.predict(X_test)
fig = plt.figure(figsize=(10,8))
plt.scatter(y_test, y_predict)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'), plt.ylabel('y predict')
plt.title('Linear regression (original), RMSE=%0.4f, R^2=%0.4f'%(mean_squared_error(y_test,y_predict),r2_score(y_test,y_predict)))
plt.grid()


# Comprobamos nuestro modelo con la base de datos unknown
unknown_original = pd.read_csv('../Dataset entrada/Examen 1/Audit_unknown.csv')
# Quitamos las mismas variables que el test
unknown = unknown_original.drop(["Unnamed: 0", "Detection_Risk"], axis=1)
# Esperamos lo mejor :)
y_examen = linreg.predict(unknown)

# Como los Audit_Risk no tienen valores negativos, los negativos los convertimos
# a 0
y_examen[y_examen < 0] = 0
# Convertimos y guardamos
unknown_original["Audit_Risk"] = y_examen
unknown_original.to_csv("Examen_pedro_mtz.csv")
































