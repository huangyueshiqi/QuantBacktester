import backtrader as bt

class MyStrategy(bt.Strategy):
    def next(self):
        print(type(self.datetime.date(0)), self.datetime.date(0))

print(hasattr(bt.linebuffer.LineBuffer, 'date'))
