import backtrader as bt
import pandas as pd
from datetime import datetime

class MyStrategy(bt.Strategy):
    def next(self):
        print(f"data0 date: {self.data0.datetime.date(0)}, strat date: {self.datetime.date()}")

cerebro = bt.Cerebro()
df1 = pd.DataFrame({'close': [10, 11]}, index=pd.to_datetime(['2020-01-01', '2020-01-02']))
df2 = pd.DataFrame({'close': [20, 21]}, index=pd.to_datetime(['2020-01-02', '2020-01-03']))
cerebro.adddata(bt.feeds.PandasData(dataname=df1), name='d1')
cerebro.adddata(bt.feeds.PandasData(dataname=df2), name='d2')
cerebro.addstrategy(MyStrategy)
cerebro.run()
