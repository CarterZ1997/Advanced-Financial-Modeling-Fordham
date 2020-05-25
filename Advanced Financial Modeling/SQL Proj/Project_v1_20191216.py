#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats
import datetime
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVR
import mysql.connector as mysql
from sqlalchemy import create_engine

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


engine = create_engine('mysql+pymysql://root:yy3529yy3529@localhost:3306/project')
con = engine.connect()


# In[2]:


data = pd.read_csv('project_data_2016_2018.csv')
#data = pd.read_sql_query('select * from project_data', con)


# In[3]:


data.drop(columns=['Unnamed: 0'], inplace=True)
data.head()


# In[4]:


data = pd.DataFrame(data)
data.columns


# In[5]:


dates = list(set(data['formatted_date']))
dates.sort()
#dates = list(map(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d"), dates))


# In[6]:


date2015D1 = "2015-01-01" #datetime.datetime.strptime('2015-01-01', '%Y-%m-%d')
date2016D1 = "2016-01-01" #datetime.datetime.strptime('2016-01-01', '%Y-%m-%d')
date2017D1 = "2017-01-01" #datetime.datetime.strptime('2017-01-01', '%Y-%m-%d')
date2018D1 = "2018-01-01" #datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')

dates2015 = list(filter(lambda x: (x >= date2015D1) & (x < date2016D1), dates))
dates2016 = list(filter(lambda x: (x >= date2016D1) & (x < date2017D1), dates))
dates2017 = list(filter(lambda x: (x >= date2017D1) & (x < date2018D1), dates))
dates2018 = list(filter(lambda x: x >= date2018D1, dates))


# In[7]:


#data2015 = data[data['formatted_date'].isin(dates2015)]
data2016 = data[data['formatted_date'].isin(dates2016)]
data2017 = data[data['formatted_date'].isin(dates2017)]
data2018 = data[data['formatted_date'].isin(dates2018)]
dataPre2018 = data[data['formatted_date'].isin(dates2015+dates2016+dates2017)]


# In[8]:


tickers = list(set(data['Ticker']))
tickers.sort()


# In[9]:


temp_df = dataPre2018[dataPre2018['Ticker'].isin(tickers[:5])]


# In[38]:


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

X = np.array(temp_df.drop(["Ticker","formatted_date","ret20_vol", "ret20_vol_tmr"], axis=1))
y = np.array(temp_df['ret20_vol_tmr'])

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1)

# Set the parameters by cross-validation
#tuned_parameters = {'kernel': ('linear', 'rbf','poly'), 
#                    'C':[0.5, 0.75, 1, 1.5, 5, 10, 100, 1000],
#                    'gamma': [1e-3, 1e-2], 
#                    'epsilon':[0.1,0.2,0.5,0.3]}

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

for score in scores:
    print("# Tuning hyper-parameters for >>>>> %s" % score.upper())
    print()
    svr = SVR()
    clf = GridSearchCV(svr, tuned_parameters, scoring="%s"%score, cv=5)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    

    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))
    #print()


    #print("The model is trained on the full development set.")
    #print("The scores are computed on the full evaluation set.")
    
    print("Score on test set: %3f" % clf.score(X_test, y_test))
    y_true, y_pred = y_test, clf.predict(X_test)
    print("Mean Absolute Error on test set: %3f" % MAE(y_test, y_pred))
    print("Mean Squared Error on test set: %3f" % MSE(y_test, y_pred))
    #print("y_true: {}\ny_pred: {}".format((y_true, y_pred)))
    print('\n\n')


# In[32]:


clf.cv_results_['mean_test_score']


# In[ ]:




