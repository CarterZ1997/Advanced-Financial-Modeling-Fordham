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