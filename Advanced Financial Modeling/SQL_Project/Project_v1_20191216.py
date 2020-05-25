#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats
import datetime
import random
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVR
import mysql.connector as mysql
from sqlalchemy import create_engine

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


engine = create_engine('mysql+pymysql://root:fordham@localhost:3306/project')
con = engine.connect()


# In[3]:


from sqlalchemy import event

def add_own_encoders(conn, cursor, query, *args):
    cursor.connection.encoders[np.float64] = lambda value, encoders: float(value)

event.listen(engine, "before_cursor_execute", add_own_encoders)


# In[4]:


#data = pd.read_csv('project_data.csv')

query = ''' select * from project_data '''

data = pd.read_sql_query(query, con)


# In[5]:


#data.drop(columns=['Unnamed: 0'], inplace=True)
data['high_1m_pct'] = data['high_1m']/data['adjclose']
data['low_1m_pct'] = data['low_1m']/data['adjclose']
data['high_5d_pct'] = data['high_5d']/data['adjclose']
data['low_5d_pct'] = data['low_5d']/data['adjclose']
data.head()


# In[6]:


data = pd.DataFrame(data)
data.columns


# In[7]:


dates = list(set(data['formatted_date']))
dates.sort()
#dates = list(map(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d"), dates))


# In[8]:


date2016D1 = "2016-01-01" #datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
date2017D1 = "2017-01-01" #datetime.datetime.strptime('2017-01-01', '%Y-%m-%d')
date2018D1 = "2018-01-01" #datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')

dates2016 = list(filter(lambda x: (x >= date2016D1) & (x < date2017D1), dates))
dates2017 = list(filter(lambda x: (x >= date2017D1) & (x < date2018D1), dates))
dates2018 = list(filter(lambda x: x >= date2018D1, dates))


# In[9]:


data2016 = data[data['formatted_date'].isin(dates2016)]
data2017 = data[data['formatted_date'].isin(dates2017)]
data2018 = data[data['formatted_date'].isin(dates2018)]
dataPre2018 = data[data['formatted_date'].isin(dates2016+dates2017)]


# In[10]:


tickers = list(set(dataPre2018['Ticker']))
train_dates = list(set(dataPre2018['formatted_date']))
train_dates.sort()

# Generate 10 different 20-day snippets for training
train_dates_subset = [train_dates[i] for i in random.sample(list(range(len(train_dates)-1)), 15)]


# In[11]:


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

params = list()
models = list()
count = 0

for date in train_dates_subset:
    curr_date = date
    next_date = train_dates[train_dates.index(curr_date)+1]    
    test_df = dataPre2018[dataPre2018['formatted_date']==next_date]        
    train_df = dataPre2018[dataPre2018['formatted_date']==curr_date]

    X_train = np.array(train_df[['daily_return', 'ret20_vol', 'high_1m_pct', 'low_1m_pct',                                  'high_5d_pct', 'low_5d_pct']])
        
    X_test = np.array(test_df[['daily_return', 'ret20_vol', 'high_1m_pct', 'low_1m_pct',                                'high_5d_pct', 'low_5d_pct']])

    y_train = np.array(train_df['ret20_vol_tmr'])
    y_test = np.array(test_df['ret20_vol_tmr'])


    # Set the parameters by cross-validation
    tuned_parameters = {'kernel': ('linear', 'rbf', 'poly'), 
                        'C':[0.1, 5, 10],
                        'gamma': [1e-1, 1], 
                        'epsilon':[0.01, 0.1, 1]}


    svr = SVR()
    clf = GridSearchCV(svr, tuned_parameters, scoring='neg_mean_squared_error', cv=5)
    clf.fit(X_train, y_train)

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Best parameters set found on development set: {}".format(clf.best_params_))
    print("Mean Absolute Error on test set: %3f" % MAE(y_test, y_pred))
    print("Mean Squared Error on test set: %3f" % MSE(y_test, y_pred))

    param = clf.best_params_
    param['MAE'] = MAE(y_test, y_pred)
    param['MSE'] = MSE(y_test, y_pred)
    param['id'] = count
    models.append(clf)
    count += 1

    params.append(param)
    
print()
print("Done")


# In[12]:


best_param_MAE = dict()
best_param_MSE = dict()
min_MSE = -1
min_MAE = -1
for param in params:
    if min_MSE < 0 and min_MAE < 0:
        min_MSE = param['MSE']
        min_MAE = param['MAE']
        best_param_MAE = best_param_MSE = param
        continue
    if param['MAE'] < min_MAE:
        min_MAE = param['MAE']
        best_param_MAE = param
    if param['MSE'] < min_MSE:
        min_MSE = param['MSE']
        best_param_MSE = param

print("Best parameter based on MAE from GridSearchCV:")
print(best_param_MAE)
print("Best parameter based on MSE from GridSearchCV:")
print(best_param_MSE)
        


# In[13]:


best_model = models[best_param_MSE["id"]]


# In[14]:


dates2017.sort()
dates2018.sort()

pred_data = data[data['formatted_date'].isin(dates2017[-1:] + dates2018)]
pred_data['prev_vol'] = pred_data['ret20_vol'].shift(1)
pred_dates = list(set(pred_data['formatted_date']))
pred_dates.sort()
n_dates = len(pred_dates)



result_df = pd.DataFrame(index=dates2018, 
                         columns=['date','prev_mean_error','pred_mean_error','prev_MSE','pred_MSE'])

for i in range(n_dates-1):
    fit_date = pred_dates[i]
    pred_date = pred_dates[i+1]
    
    fit_df = pred_data[pred_data['formatted_date']==fit_date]
    test_df = pred_data[pred_data['formatted_date']==pred_date]
    common_stocks = set(fit_df['Ticker']).intersection(set(test_df['Ticker']))
    fit_df = fit_df[fit_df['Ticker'].isin(common_stocks)]
    test_df = test_df[test_df['Ticker'].isin(common_stocks)]
    X = np.array(fit_df[['daily_return', 'ret20_vol', 'high_1m_pct', 'low_1m_pct', 'high_5d_pct', 'low_5d_pct']])
    y_true = np.array(fit_df[['ret20_vol_tmr']])
    y_prev = np.array(fit_df[['ret20_vol']])
    y_pred = best_model.predict(X)
    prev_avg_error = np.mean(y_prev - y_true)
    pred_avg_error = np.mean(y_pred - y_true)
    prev_MSE = MSE(y_true, y_prev)
    pred_MSE = MSE(y_true, y_pred)
    
    result_df.loc[pred_date] = [pred_date, prev_avg_error, pred_avg_error, prev_MSE, pred_MSE]

result_df.index = list(range(len(result_df)))
display(result_df)


# In[16]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(18, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot( 'date', 'prev_mean_error', data=result_df)
plt.plot( 'date', 'pred_mean_error', data=result_df)
plt.legend()
plt.show()


# In[18]:


## drop_table_cmd = '''IF OBJECT_ID('Result') IS NOT NULL DROP TABLE Result; '''
# con.execute(drop_table_cmd)

#create_table_cmd = '''
#CREATE TABLE Result 
#(
#  date VARCHAR(10) NOT NULL,
#  PrevRealVol_Error FLOAT NULL,
#  PredVol_Error FLOAT NULL,
#  PrevRealVol_MSE FLOAT NULL,
#  PredVol_MSE FLOAT NULL,
#  PRIMARY KEY (date)
#)
#'''

#con.execute(create_table_cmd)

result_df.to_sql('Result', con=engine, index=False, if_exists = 'replace')


# In[ ]:




