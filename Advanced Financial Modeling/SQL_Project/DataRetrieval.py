#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials
import requests
import mysql.connector as mysql

from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://root:formham@localhost:3306/project')
con = engine.connect()


symbol = pd.read_csv("symbol.csv")
tickers = list(symbol['Ticker'])
tickers.sort()
start_date = '2016-01-01'
end_date = '2018-12-31'
method = 'daily'

yahoo = YahooFinancials('AAPL')
x = yahoo.get_historical_price_data(start_date, end_date, method)
x['AAPL'].keys()


df = pd.DataFrame()
i = 0
for ticker in tickers:
    ticker = symbol.at[i,'Ticker']
    #check if ticker is valid
    link = requests.get('http://finance.yahoo.com/quote/' + ticker)
    if link.url.find("lookup") < 0:
        yahoo = YahooFinancials(ticker)
        price = yahoo.get_historical_price_data(start_date,end_date,method)
        #check if ticker has data
        if len(price[ticker])==6:
            #check if ticker has full historical daily prices between 2016 and 2018
            if (len(price[ticker]["prices"]) > 600): # 80% of total trading days in 2016-2018
                prices = pd.DataFrame(price[ticker]['prices'])                
                n = len(prices)
                prices['daily_return'] = [np.log(prices['adjclose'][i]/prices['adjclose'][i-1]) if i > 0 else np.nan for i in range(n)]
                prices['ret20_vol'] = [np.std(prices['daily_return'][(i-19):(i+1)],ddof=19) if (i >= 20) and (i+1 < n) else np.nan for i in range(n)]
                prices['ret20_vol_tmr'] = prices['ret20_vol'].shift(-1)
                prices['next_ret20_vol'] = [np.std(prices['daily_return'][(i+1):(i+21)], ddof=19) if (i >= 0) and (i+21) < n else np.nan for i in range(n)]
                prices['high_1m'] = [np.max(prices['adjclose'][(i-19):i]) if i >= 19 else np.nan for i in range(n)]
                prices['low_1m'] = [np.min(prices['adjclose'][(i-19):i]) if i >= 19 else np.nan for i in range(n)]
                prices['high_5d'] = [np.max(prices['adjclose'][(i-4):i]) if i >= 4 else np.nan for i in range(n)]
                prices['low_5d'] = [np.max(prices['adjclose'][(i-4):i]) if i >= 4 else np.nan for i in range(n)]
                prices = prices.drop(columns=['date'])
                prices['Ticker'] = ticker
                df = df.append(prices[['Ticker', 'formatted_date', 'adjclose', 'daily_return',                                       'ret20_vol', 'ret20_vol_tmr', 'next_ret20_vol',                                        'high_1m', 'low_1m', 'high_5d', 'low_5d']], ignore_index = True)
                
    i += 1
    if (i%10 == 0):
        print(ticker)

df.to_excel("2016_2018_daily_data_v2.xlsx", index=False)
df.to_csv("2016_2018_daily_data_v2.csv", index=False)
print("\nDone.")

df


# In[ ]:




