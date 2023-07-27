#%%

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Modeling Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,r2_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#%% Importing data
Train = pd.read_csv('Audit_train.csv', index_col=0)
Test = pd.read_csv('Audit_test.csv', index_col=0)
Unknown = pd.read_csv('Audit_unknown.csv', index_col=0)
#%% remove detection_risk

Train = Train.drop(columns=['Detection_Risk'])

#%% remove outliers
Train = Train[(np.abs(stats.zscore(Train)) < 3).all(axis=1)]

#%% scale train
sc_X = MinMaxScaler()
sc_X.fit(Train)
Train = sc_X.transform(Train)
#%%
Train = Train + 1
#%% data transformation
Train_flat = []

for cols in range(Train.shape[1]):
    try:
        fitted_data, _ = stats.boxcox(Train[:,cols])
        Train_flat.append(list(fitted_data))
    except:
        Train_flat.append(Train[:,cols])

Train_flat = np.array(Train_flat)
Train = Train_flat.reshape(Train.shape)
#%% linear regression
X_train = Train[:, 0:23]
y_train = Train[:, 23]

linreg = LinearRegression()
linreg.fit(X_train, y_train)


#%% test transform pipeline
Test = Test.drop(columns=['Detection_Risk'])
#Test = Test[(np.abs(stats.zscore(Test)) < 3).all(axis=1)]
Test = sc_X.transform(Test)
Test = Test + 1
Test_flat = []

for cols in range(Test.shape[1]):
    try:
        fitted_data, _ = stats.boxcox(Test[:,cols])
        Test_flat.append(list(fitted_data))
    except:
        Test_flat.append(Test[:,cols])

Test_flat = np.array(Test_flat)
Test = Test_flat.reshape(Test.shape)


#%%
X_test = Test[:, 0:23]
y_test = Test[:, 23]

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

#%% unknown dataset
Unknown = Unknown.drop(columns=['Detection_Risk'])
sc_X = MinMaxScaler()
sc_X.fit(Unknown)
Unknown = sc_X.transform(Unknown)
Unknown = Unknown + 1
Unknown_flat = []

for cols in range(Unknown.shape[1]):
    try:
        fitted_data, _ = stats.boxcox(Unknown[:,cols])
        Unknown_flat.append(list(fitted_data))
    except:
        Unknown_flat.append(Unknown[:,cols])

Unknown_flat = np.array(Unknown_flat)
Unknown = Unknown_flat.reshape(Unknown.shape)

y_predict_u = linreg.predict(Unknown)
# %%
Unknown_or = pd.read_csv('Audit_unknown.csv', index_col=0)
Unknown_or['Audit_Risk'] = y_predict_u
Unknown_or.to_csv('Unknown_dataset_output.csv')