import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

class DividendHandler:
    """
    分红处理组件
    
    负责处理现金分红、送股和拆股等操作
    """
    
    def __init__(self, strategy, dividends, dividends_probonus, dividends_changert, 
                 portfolio, params):
        """
        初始化分红处理组件
        
        参数:
            strategy: 策略对象，用于访问broker和数据
            dividends: 现金分红数据
            dividends_probonus: 送股数据
            dividends_changert: 拆股数据
            portfolio: Portfolio对象，统一管理持仓状态
            params: 策略参数
        """
        self.strategy = strategy
        self.dividends = dividends
        self.dividends_probonus = dividends_probonus
        self.dividends_changert = dividends_changert
        self.portfolio=portfolio
        self.params = params
        self.order_list = []
    
    def deal_dividend(self, cur_date):
        """
        处理分红+送股+拆股
        
        参数:
            cur_date: 当前日期
        """
        if not self.params.deal_dividend:
            return  # 如果不处理分红，直接返回
        
        # 对所持仓位进行分红处理
        for stock in self.portfolio.get_holding_stocks():
            stock_positions=self.portfolio.get_stock_positions(stock)
            for buy_date in sorted(stock_positions.keys()):
                # 处理现金分红
                self._handle_cash_dividend(stock, buy_date, cur_date)
                
                # 处理送股
                self._handle_stock_bonus(stock, buy_date, cur_date)
                
                # 处理拆股
                self._handle_stock_split(stock, buy_date, cur_date)
    
    def _handle_cash_dividend(self, stock, buy_date, cur_date):
        """
        处理现金分红
        
        参数:
            stock: 股票代码
            buy_date: 买入日期
            cur_date: 当前日期
        """
        if stock in self.dividends:
            # 记录哪些仓位是有可能需要扣分红税的
            for regdate in self.dividends[stock]:
                # 符合分红登记条件且当前日期为分红到账日期
                if (buy_date < regdate
                        and self.dividends[stock][regdate]['status'] == 'registration'
                        and self.dividends[stock][regdate]['cashdate'] == cur_date):
                    # 处理分红到账,计算分红金额
                    stock_positions=self.portfolio.get_stock_positions(stock)
                    position_size=stock_positions[buy_date].size
                    cash_per_share = self.dividends[stock][regdate]['cash']
                    dividend_amount = position_size * cash_per_share
                    dividend_amount = round(dividend_amount, 8)
                    print(f'现金分红持仓{position_size},每股金额{cash_per_share},总金额{dividend_amount}')
                    
                    # 更新账户现金
                    self.strategy.broker.add_cash(dividend_amount)  # 现金发放日操作
                    # 更新分红状态
                    self.dividends[stock][regdate]['status'] = 'dividend_payment'

                    # 更新持仓对应的现金分红金额记录
                    self.portfolio.add_dividend_cash_by_date(stock,buy_date,dividend_amount)
                    print(f'现金分红股票{stock},购买日期{buy_date},持仓对应的分红金额{dividend_amount}')
                    
                    # 现金分红填权
                    if self.params.cash2shares:
                        self._reinvest_cash_dividend(stock, dividend_amount)
    
    def _reinvest_cash_dividend(self, stock, dividend_amount):
        """
        将现金分红再投资（填权）
        受保护方法，不推荐外部访问
        参数:
            stock: 股票代码
            dividend_amount: 分红金额
        """
        buy_data = self.strategy.getdatabyname(stock)
        close_price = buy_data.close[0]
        buy_unit = math.floor(dividend_amount / close_price / 100) * 100
        
        if buy_unit > 0:
            order = self.strategy.buy(data=buy_data, size=buy_unit)
            self.order_list.append(order)
            # 分红没有手续费，这里需要添加手续费到现金中
            handing_fee = close_price * (1 + self.strategy.perc) * buy_unit * self.strategy.commission
            self.strategy.broker.add_cash(handing_fee)
            print(f'现金分红填权：股票{stock},当前股价{close_price},买入股数{buy_unit},添加的手续费{handing_fee}')
    
    def _handle_stock_bonus(self, stock, buy_date, cur_date):
        """
        处理送股
        
        参数:
            stock: 股票代码
            buy_date: 买入日期
            cur_date: 当前日期
        """
        if stock in self.dividends_probonus:
            for regdate in self.dividends_probonus[stock]:
                # 股份到账日和当前日期匹配
                if (buy_date < regdate
                        and self.dividends_probonus[stock][regdate]['status'] == 'registration'
                        and self.dividends_probonus[stock][regdate]['sharedate'] == cur_date):
                    
                    # 更新送股状态
                    self.dividends_probonus[stock][regdate]['status'] = 'dividend_payment'
                    # 计算注册日期之前的总持仓量
                    total_size = 0
                    stock_positions=self.portfolio.get_stock_positions(stock)
                    for buy_dt in sorted(stock_positions.keys()):
                        if buy_dt < regdate:
                            total_size += stock_positions[buy_dt].size
                    
                    # 送股操作
                    if self.dividends_probonus[stock][regdate]['probonus'] > 0 and total_size > 0:
                        stock_data = self.strategy.getdatabyname(stock)
                        # 送股产生的现金价值
                        bonus2cash = self.dividends_probonus[stock][regdate]['probonus'] * total_size * \
                                     stock_data.close[0]
                        # 送股总数量
                        bonus = math.floor(self.dividends_probonus[stock][regdate]['probonus'] * total_size)
                        
                        # 向账户添加现金
                        self.strategy.broker.add_cash(bonus2cash)
                        order = self.strategy.buy(data=stock_data, size=bonus)
                        self.order_list.append(order)
                        # 分红没有手续费，这里需要添加手续费到现金中
                        close_price = stock_data.close[0]
                        handing_fee = close_price * (1 + self.strategy.perc) * bonus * self.strategy.commission
                        self.strategy.broker.add_cash(handing_fee)
                        print(f'送股操作：股票{stock},送股产生的现金价值{bonus2cash},送股总数量{bonus},添加的手续费{handing_fee}')
    
    def _handle_stock_split(self, stock, buy_date, cur_date):
        """
        处理拆股
        
        参数:
            stock: 股票代码
            buy_date: 买入日期
            cur_date: 当前日期
        """
        if stock in self.dividends_changert:
            for regdate in self.dividends_changert[stock]:
                # 上市日和当前日期匹配
                if (buy_date < regdate
                        and self.dividends_changert[stock][regdate]['status'] == 'registration'
                        and self.dividends_changert[stock][regdate]['listdate'] == cur_date):
                    # 更新拆股状态
                    self.dividends_changert[stock][regdate]['status'] = 'dividend_payment'
                    # 获取当前股票数据和持仓
                    stock_data = self.strategy.getdatabyname(stock)
                    curPosition = self.strategy.getposition(stock_data).size
                    # 计算新的持仓数量
                    newPosition = math.floor(curPosition * self.dividends_changert[stock][regdate]['ratio'])
                    newPositionCash = curPosition * self.dividends_changert[stock][regdate]['ratio'] * stock_data.close[0]
                    # 向账户添加现金
                    self.strategy.broker.add_cash(newPositionCash)
                    # 根据现金购买更多股票
                    buy_unit = math.floor(newPositionCash / stock_data.close[0] / 100) * 100
                    order = self.strategy.buy(data=stock_data, size=buy_unit)
                    self.order_list.append(order)
                    # 分红没有手续费，这里需要添加手续费到现金中
                    close_price = stock_data.close[0]
                    handing_fee = close_price * (1 + self.strategy.perc) * buy_unit * self.strategy.commission
                    self.strategy.broker.add_cash(handing_fee)
                    print(f'拆股操作：股票{stock},拆股比例{self.dividends_changert[stock][regdate]["ratio"]},新持仓{newPosition},当前股价{stock_data.close[0]},买入股数{buy_unit}')
    
    def calculate_tax(self, buy_date, sell_date):
        """
        计算持股周期对应的扣税系数
        
        参数:
            buy_date: 购买日期
            sell_date: 卖出日期
            
        返回:
            扣税系数
        """
        buy_datetime = datetime.strptime(buy_date, '%Y-%m-%d')
        sell_datetime = datetime.strptime(sell_date, '%Y-%m-%d')
        
        # 计算持有月数
        delta = relativedelta(sell_datetime, buy_datetime)
        holding_months = delta.years * 12 + delta.months
        
        # 根据持有时间确定税率
        if holding_months <= 1:
            return 0.2
        elif holding_months <= 12:
            return 0.1
        else:
            return 0
    
    def get_dividend_tax(self, stock, size, current_date):
        """
        先进先出卖出股票，并在对有分红登记的股票，计算卖出该股票时对应的现金分红扣税金额
        
        参数:
            stock: 股票代码
            size: 卖出数量
            current_date: 当前日期
            
        返回:
            现金分红扣税金额
        """
        # ！！注意：backtrader中卖出股票的order.executed.size是负数
        if size < 0:
            size = 0 - size
        
        # 检查总持仓是否足够卖出
        total_size = self.portfolio.get_stock_total_size(stock)

        if size > total_size:
            print(f'警告:股票{stock}卖出数量大于总持仓({size}>{total_size})')
            return 0
        
        # 计算分红扣税
        remaining_size = size  # 记录需要卖出的股票数量
        dividendTax = 0  # 累计计算分红税

        stock_positions=self.portfolio.get_stock_positions(stock)
        for buy_date in sorted(stock_positions.keys()):
            position = stock_positions[buy_date]
            tax_rate=0
            # 检查是否有分红记录
            if position.dividend_cash>0:
                tax_rate = self.calculate_tax(buy_date, current_date.strftime('%Y-%m-%d'))
                print(f'股票{stock},分红扣税税率{tax_rate},买入日期{buy_date},当前日期{current_date}')
            
            position_size = position.size
            if position_size > 0:
                if position_size > remaining_size:
                    # 如果当前股票持仓数量大于剩余卖出数量，则只扣减剩余卖出数量，并退出循环(部分卖出：当前批次足够覆盖剩余卖出量)
                    if position.dividend_cash>0:
                        dividendTax+=tax_rate*position.dividend_cash*(remaining_size/position_size)
                    break
                else:
                    # 当前股票持仓数量小于剩余卖出数量，则扣减当前股票持仓数量，并继续循环(全部卖出：当前批次全部卖完，继续下一批次)
                    remaining_size -= position_size
                    if position.dividend_cash>0:
                        #根据仓位数量计算分红扣税
                        dividendTax+=tax_rate*position.dividend_cash
        
        # 注意：实际的持仓更新将在Portfolio.remove_position中处理
        
        return dividendTax 