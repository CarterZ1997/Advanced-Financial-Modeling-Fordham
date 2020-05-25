# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from yahoofinancials import YahooFinancials

ticker = 'AAPL'

yahoo = YahooFinancials(ticker)

yahoo.get_historical_price_data('2018-01-01', '2018-01-10', 'daily')


import pandas as pd

data = pd.DataFrame(yahoo.get_historical_price_data('2018-01-01', '2018-01-10', 'daily')['AAPL']['prices'])

data.head(10)

data.to_




import numpy as np

from sklearn.svm import SVR

x = np.random.rand(40,1)

y = np.sin(x).ravel()

svr_linear = SVR('linear', C=100, gamma = 'auto')
        
regression_form = svr_linear.fit(x,y)

x_1 = np.random.rand(1,1)

regression_form.predict(x_1)

x_2 = np.random.rand(3,1)

regression_form.predict(x_2)

import pandas as pd

pd.DataFrame(regression_form.predict(x_1))

