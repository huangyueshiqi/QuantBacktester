import math

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.config import config
from utils.helpers import cvxopt, get_stock_dic, get_st_stock

# 导入新创建的组件
from core.dividend_handler import DividendHandler
from core.trade_executor import TradeExecutor
from core.state_manager import StateManager
from core.benchmark_calculator import BenchmarkCalculator
from core.position import Portfolio
from core.constraint_manager import create_constraint_manager_from_params


class BuyStrategy(bt.Strategy):
    """
    Buy选股策略
    """
    params=()

    def __init__(self, st_dict, dividends, dividends_probonus, dividends_changert, BenchmarkDetailData,
                 BaseStockDetailData, start_date,end_date,cash, commission, perc, codeslist,buy_list):
        """
            # 初始化函数
            # 参数：
            # stockSTData: ST股票数据，包括股票代码等
            # stockDividentData: 股票分红数据，包括股票代码，分红日期，分红金额等
            # stockProrightData: 股票配股数据，包括股票代码，配股数量等
            # BenchmarkDetailData: 基准指数数据，包括['TRADEDATE', 'TCLOSE']，日期和指数收盘数值
            # BaseStockDetailData: 基准成分股票数据，包括'TRADEDATE', 'STOCKCODE', 'WEIGHT', 'MARKETVALUE', 'TYPE', 'INDCODE'
            # startDate 起始日期
            # endDate 结束日期
            # codes 股票列表
            # cash 起始现金金额
            # adjust_data 调仓表数据

        """
        #动态设置参数
        self.params=type('Params',(),{
            'period': config.backtest.strategy.period,  # 默认周期
            'execution_price': config.backtest.strategy.execution_price,  # 默认为第二天开盘价
            'score_threshold': config.backtest.strategy.score_threshold,  # 分数阈值
            'topk': config.backtest.strategy.topk,  # topN取值,默认为100
            'low_stopping': config.backtest.strategy.low_stopping,  # 离跌停价的2个点的距离内默认为跌停,卖不出
            'high_limit': config.backtest.strategy.high_limit,  # 离涨停价的2个点的距离内默认为涨停,买不到
            'vol_percent': config.backtest.strategy.vol_percent,  # 日成交量的比例,高于该比例买卖得不到执行
            'max_position': config.backtest.strategy.max_position,  # 占资产总最大持仓比例
            'deal_dividend': config.backtest.strategy.deal_dividend,  # 是否处理分红,True 处理;否则,False
            'cash2shares': config.backtest.strategy.cash2shares,  # 现金分红是否填权，填权为True;否则,False
            'round_to_hundred': config.backtest.strategy.round_to_hundred,  # 是否将股票数量设置为100的整数倍
        })()

        self.addminperiod(self.params.period)  # 最小周期数
        self.start_date=start_date
        self.end_date=end_date

        self.order = None

        self.order_list = []  # 记录以往订单，方便调仓日对未完成订单做处理
        self.cash=cash
        self.commission = commission
        self.perc = perc

        #统一的持仓管理器
        self.portfolio=Portfolio()
        self.dividends = dividends  # 记录分红事件
        self.dividends_probonus = dividends_probonus  # 记录送股事件
        self.dividends_changert = dividends_changert  # 记录拆股事件

        self.total_profit = 0  # 总的利润
        self.total_trades = 0  # 总的买卖交易次数
        self.winning_trades = 0  # 与 total_trades 计算胜率

        # 计算盈亏比
        self.reward = 0.0  # 交易盈利数
        self.rewardcnt = 0  # 交易盈利次数
        self.risk = 0.0  # 交易亏损数
        self.riskcnt = 0  # 交易亏损次数

        # 每日资产总额数据
        self.value = []
        self.cash_history=[]
        self.position_value_history=[]
        self.dates=[]

        self.total_commission = 0  # 总的佣金

        # st数据
        self.st_dict = st_dict

        # benchmark数据
        self.BenchmarkDetailData = BenchmarkDetailData
        # benchmark stock数据
        self.BaseStockDetailData = BaseStockDetailData


        # 初始化组件
        self.benchmark_calculator = BenchmarkCalculator(
            BenchmarkDetailData, start_date, end_date
        )
        self.benchmark_portfolio_daily = self.benchmark_calculator.get_benchmark_portfolio_daily()
        self.benchmark_annualReturn = self.benchmark_calculator.get_benchmark_annual_return()

        self.dividend_handler = DividendHandler(
            self, dividends, dividends_probonus, dividends_changert,
            self.portfolio, self.params
        )

        self.trade_executor = TradeExecutor(self, self.params)
        self.state_manager = StateManager(self, self.params)

        #将股票池列表转换为集合
        self.stock_pool = set(codeslist) if codeslist else set()

        # 新增参数
        self.buy_list = buy_list
        self.buy_signal_dict={}
        if self.buy_list is not None:
            for date,group in self.buy_list.groupby(self.buy_list['datetime'].dt.strftime("%Y-%m-%d")):
                self.buy_signal_dict[date]=group['instrument'].tolist()
        self.holding_periods = 40  # 持仓天数
        self.take_profit = 0.5
        self.stop_loss = 0.3
        self.order_reasons = {}
        self.verbose=False

        # 添加交易统计相关变量
        self.trade_stats = []  # 存储交易统计数据

        # 资金管理变量
        self.total_days = 40  # 总份数
        self.accumulated_cash = 0  # 累积资金

        # 添加技术指标字典
        self.indicators = {}
        self.data_dict={}

        for data in self.datas:
            self.data_dict[data._name]=data
            self.indicators[data._name] = {
                'max_30': bt.indicators.Highest(data.close, period=30, plot=False),
                'min_30': bt.indicators.Lowest(data.close, period=30, plot=False),
                'mean_30': bt.indicators.SMA(data.close, period=30, plot=False),
                'std_30': bt.indicators.StdDev(data.close, period=30, plot=False),
                # 'sma_10': bt.indicators.SMA(data.close, period=10, plot=False),
                # 'sma_20': bt.indicators.SMA(data.close, period=20, plot=False),
                # 'sma_30': bt.indicators.SMA(data.close, period=30, plot=False),
            }



    def log(self, txt, dt=None):
        """
        日志函数

        参数:
            txt: 日志文本
            dt: 日期时间
        """
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def log_info(self,message):
        if self.verbose:
            print(message)

    def get_position_value(self):
        """
        计算当前所有持仓股票的总价值

        返回:
            当前所有持仓股票的总价值
        """
        position_value = 0
        for stock in self.portfolio.get_holding_stocks():
            stock_data = self.getdatabyname(stock)
            position_value += self.getposition(stock_data).size * stock_data.close[0]
        return position_value


    def cancel_pending_orders(self):
        """
        取消所有的订单
        """
        if len(self.order_list) > 0:
            for order in self.order_list:
                self.log_info(f"订单信息：股票:{order.data._name},存活状态{order.alive()},{order.getstatusname()},类型：{order.ordtypename()}")
                self.cancel(order)  # 如果订单未完成，则撤销订单
            self.order_list = []  # 重置订单列表



    def prenext(self):
        """在策略正式开始前的预处理阶段调用"""
        self.next()

    def next(self):
        """
        主要策略逻辑
        每个bar都会调用这个方法
        """
        # cur_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        # print(f"cur_date:{cur_date}")
        #
        # # 找到截至当前日期的最近日期的ST股票列表
        # cur_st_list = get_st_stock(self.st_dict, cur_date)
        #
        # # 卖出st持仓股票
        # self.trade_executor.sell_st_stocks(cur_st_list)
        #
        # # 分红处理
        # # len(self.data)为已处理数据的长度,self.data._idx为数据总长度,最后一天不处理分红
        # if len(self.data) - self.data._idx != 0:
        #     self.dividend_handler.deal_dividend(cur_date)

        # 调仓日处理
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        self.cancel_pending_orders()

        # 获取当前持仓
        # 检查是否有需要卖出的股票（持仓超过天数限制或触发止盈止损）
        stocks_to_sell = []
        self.log_info(f"{current_date},当前持仓：{self.portfolio.get_holding_stocks()}")
        for stock_name in self.portfolio.get_holding_stocks():
            data = self.data_dict.get(stock_name)
            current_price = data.close[0]
            stock_positions=self.portfolio.get_stock_positions(stock_name)
            earliest_buy_date=self.portfolio.get_stock_buy_date(stock_name)

            # if stock_name == '002595.SZ':
            #     aliases = data.lines.getlinealiases()
            #     print(f"line aliases: {aliases}")
            #
            #     # 打印每个字段对应的当前值
            #     values = [getattr(data, alias)[0] for alias in aliases]
            #     print(f"line values: {values}")


            #停牌检查
            if data.tradestatus[0] == 0:
                print(f'在{current_date}卖出股票{stock_name}停牌')
                continue

            if earliest_buy_date:
                earliest_pos=stock_positions[earliest_buy_date]
                buy_price = earliest_pos.price

                # 计算收益率
                profit_loss_ratio = (current_price - buy_price) / buy_price

                # 1. 检查是否达到持仓天数上限 (使用Portfolio的方法)
                sell_reason=None
                if self.portfolio.can_sell_stock(stock_name, current_date, self.holding_periods):
                    holding_days = self.portfolio.get_holding_days(stock_name, current_date)
                    sell_reason = 'holding_period'
                    print(f"股票 {stock_name} 已持仓 {holding_days} 天，准备卖出")

                # 2. 检查是否触发止盈（盈利30%）
                elif profit_loss_ratio >= self.take_profit:
                    sell_reason = 'take_profit'
                    print(
                        f"股票 {stock_name} 盈利 {profit_loss_ratio:.2%}，达到止盈条件 {self.take_profit:.0%}，准备卖出")

                # 3. 检查是否触发止损（亏损10%）
                elif profit_loss_ratio <= -self.stop_loss:
                    sell_reason = 'stop_loss'
                    print(
                        f"股票 {stock_name} 亏损 {profit_loss_ratio:.2%}，达到止损条件 {self.stop_loss:.0%}，准备卖出")

                # 4. 检查是否在卖出列表中
                # elif stock_name in self.sell_list:
                #     sell_reason = 'sell_list'

                if sell_reason:
                    # 获取总持仓量
                    total_size = self.portfolio.get_stock_total_size(stock_name)

                    stocks_to_sell.append({
                        'name': stock_name,
                        'data': data,
                        'size': total_size,
                        'price': buy_price,  # 记录触发条件时的参考买入价
                        'reason': sell_reason,
                        'buy_date': earliest_buy_date  # 记录最早买入日期作为参考
                    })

        # 执行卖出操作
        for sell_info in stocks_to_sell:
            data = sell_info['data']
            size_to_sell = sell_info['size']
            sell_reason = sell_info['reason']

            # 在Backtrader下单
            order = self.sell(data=data, size=size_to_sell)
            self.order_list.append(order)
            if order:
                self.order_reasons[order.ref]=sell_reason

        # 检查今天是否有需要买入的股票
        # 获取当前持仓的股票代码集合 (Portfolio)
        current_holding_codes = self.portfolio.get_holding_stocks()

        # today_buy_signals = self.buy_list[self.buy_list['datetime'] == pd.to_datetime(current_date)]
        today_buy_instruments=self.buy_signal_dict.get(current_date,[])

        # 确定需要买入的股票（在信号列表中但不在当前持仓中）
        stocks_to_buy = []
        for stock_name in today_buy_instruments:
            # 检查是否已经持仓
            if stock_name not in current_holding_codes:
                # 查找对应的数据源
                data=self.data_dict.get(stock_name)
                if data:
                    try:
                        stock_price = data.close[0]
                        # sma_20 = self.indicators[stock_name]['sma_20'][0]
                        # sma_30 = self.indicators[stock_name]['sma_30'][0]
                        # if stock_price < sma_20 *1.05 and sma_20 < sma_30:
                        #     stocks_to_buy.append({
                        #         'name': stock_name,
                        #         'data': data,
                        #         'close': data.close[0]
                        #     })
                        #     break
                        max_30 = self.indicators[stock_name]['max_30'][0]
                        min_30 = self.indicators[stock_name]['min_30'][0]
                        mean_30 = self.indicators[stock_name]['mean_30'][0]
                        std_30 = self.indicators[stock_name]['std_30'][0]

                        should_buy=False
                        if stock_price < max_30 * 0.95 and (max_30 - min_30) / mean_30 < 0.2 and std_30 / mean_30 < 0.3:
                            should_buy=True
                        if should_buy:
                            stocks_to_buy.append({
                                'name': stock_name,
                                'data': data,
                                'close': data.close[0]
                            })
                    except Exception as e:
                        print(f"警告：{stock_name}:{e}")
                        pass

        # 执行买入操作
        # 计算当天可用资金（初始资金的1/60加上累积资金）
        daily_cash = self.broker.startingcash / self.total_days
        pre_cash = daily_cash + self.accumulated_cash
        current_cash=self.broker.getcash()
        current_value=self.broker.getvalue()
        available_cash=min(current_cash,pre_cash)
        self.log_info(f"daily_cash:{daily_cash},pre_cash:{pre_cash},available_cash:{available_cash}")

        if stocks_to_buy and available_cash > 0:
            # 计算每只股票分配的资金（平均分配）
            cash_per_stock = available_cash / len(stocks_to_buy)
            if cash_per_stock>current_value*0.1:
                cash_per_stock=current_value*0.1

            for stock in stocks_to_buy:
                stock_name = stock['name']
                data = stock['data']
                stock_price = stock['close']

                if data.tradestatus[0] == 0:
                    print(f'在{current_date}买入股票{stock_name}停牌')
                    continue

                # 计算购买数量
                if cash_per_stock > 0 and stock_price > 0:
                    # 使用95%的资金以防计算误差
                    size = int((cash_per_stock * 0.95) / stock_price/100)*100

                    if size > 0:
                        buy_value = size * stock_price

                        # 打印买入订单的详细信息
                        self.log_info("=" * 60)
                        self.log_info(f"买入订单详情:")
                        self.log_info(f"  股票代码: {stock_name}")
                        self.log_info(f"  买入日期: {current_date}")
                        self.log_info(f"  买入价格: {stock_price:.3f}")
                        self.log_info(f"  买入数量: {size}")
                        self.log_info(f"  买入金额: {buy_value:.2f}")
                        self.log_info("=" * 60)

                        order=self.buy(data=data, size=size)
                        self.order_list.append(order)

                        # 减去已使用的资金
                        available_cash -= buy_value
                    else:
                        print(f"警告：资金不足，无法购买 {stock_name}")
                else:
                    print(f"警告：价格异常或资金不足，无法购买 {stock_name}")

            # 更新累积资金
            self.accumulated_cash = available_cash
        else:
            # 没有股票可买，将当天资金累积到下一天
            self.accumulated_cash += daily_cash
            print(f"今日无股票可买，资金累积至下次交易，累积资金: {self.accumulated_cash:.2f}")



    def notify_order(self, order):
        """
        处理订单通知

        参数:
            order: 订单对象
        """
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态 submitted/accepted，无动作
            return

        # 订单完成
        if order.status in [order.Completed, order.Partial]:
            if order.isbuy():
                # 记录建仓时间和数量
                stock = order.data._name
                str_buy_date = bt.num2date(order.executed.dt).strftime('%Y-%m-%d')

                #使用Portfolio对象记录持仓
                self.portfolio.add_position(stock,str_buy_date,order.executed.size,order.executed.price)

                self.log(
                    f'买单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
            elif order.issell():
                # 卖出股票，扣税, 先进先出调整头寸
                stock = order.data._name
                current_date = bt.num2date(order.executed.dt)

                tax = self.dividend_handler.get_dividend_tax(stock, order.executed.size, current_date)

                # 使用Portfolio对象处理卖出（FIFO原则）
                sold_positions=self.portfolio.remove_position(stock, order.executed.size)

                sell_reason=self.order_reasons.pop(order.ref,'unknown')

                # 记录交易统计信息 (可能有多笔，因为FIFO)
                for sold_pos in sold_positions:
                    buy_price = sold_pos.price
                    buy_date = sold_pos.buy_date

                    # 计算该笔交易的实际收益率
                    profit_loss_ratio = (order.executed.price - buy_price) / buy_price
                    #计算绝对收益金额(忽略佣金)
                    profit_loss_amount= (order.executed.price - buy_price) * sold_pos.size

                    # 打印卖出订单的详细信息
                    self.log_info("=" * 60)
                    self.log_info(f"卖出订单详情:")
                    self.log_info(f"  股票代码: {stock}")
                    self.log_info(f"  买入日期: {buy_date}")
                    self.log_info(f"  买入价格: {buy_price:.3f}")
                    self.log_info(f"  卖出日期: {current_date}")
                    self.log_info(f"  卖出价格: {order.executed.price:.3f}")
                    self.log_info(f"  收益率: {profit_loss_ratio:.2%}")
                    self.log_info(f"  卖出原因: {sell_reason}")
                    self.log_info(f"  持仓数量: {sold_pos.size}")
                    self.log_info("=" * 60)

                    self.trade_stats.append({
                        'stock': stock,
                        'buy_date': buy_date,
                        'sell_date': current_date,
                        'buy_price': buy_price,
                        'sell_price': order.executed.price,
                        'profit_loss_ratio': profit_loss_ratio,
                        'profit_loss_amount': profit_loss_amount,
                        'is_win': profit_loss_ratio > 0,
                        'sell_reason': sell_reason
                    })



                if self.params.deal_dividend and tax != 0:
                    tax_date = current_date.strftime('%Y-%m-%d')
                    self.log(f'在{tax_date}股票{stock}分红扣税:{tax}')
                    self.broker.add_cash(0 - tax)
                    # 增量回测时分红扣税处理
                    if not self.dividend_tax:
                        self.dividend_tax = {tax_date: tax}
                    else:
                        # 取出已保存的最新日期
                        current_date = list(self.dividend_tax.keys())[0]
                        if tax_date > current_date:
                            # 新订单日期更新,覆盖已有记录
                            self.dividend_tax = {tax_date: tax}
                        elif tax_date == current_date:
                            # 同一天累加扣税金额
                            self.dividend_tax[tax_date] += tax

                self.log(
                    f'卖单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
        else:
            self.log(
                f'警告：订单作废  {order.data._name}, {order.getstatusname()}, isbuy:{order.isbuy()}, {order.created.size}, 创建时间 {bt.num2date(order.created.dt)}')
            # 检查是否因为余额不足
            cash = self.broker.getcash()
            total_cost = order.size * order.data.close[0]
            self.log(f'Order Size:{order.size},Price:{order.data.close[0]},Total Cost:{total_cost},Available Cash:{cash}')
            if total_cost > cash:
                self.log('Confirmed:Margin due to insufficient balance')

    def notify_trade(self, trade):
        """
        记录交易收益情况

        参数:
            trade: 交易对象
        """
        if trade.isclosed:  # 买&卖，交易结束
            self.reward += (trade.pnlcomm > 0) * trade.pnlcomm  # 盈利交易里总盈利金额
            self.rewardcnt += (trade.pnlcomm > 0)  # 盈利交易次数
            self.risk += (trade.pnlcomm < 0) * (0 - trade.pnlcomm)  # 亏损交易里总的亏损金额
            self.riskcnt += (trade.pnlcomm < 0)  # 亏损交易次数
            risk_reward_ratio = 0
            if self.rewardcnt * self.riskcnt:
                # 盈亏比
                risk_reward_ratio = (self.reward / self.rewardcnt) / (self.risk / self.riskcnt)

            self.total_profit += trade.pnlcomm  # 总的利润
            self.total_trades += 1  # 总的交易次数
            self.winning_trades += (trade.pnlcomm > 0)  # 盈利交易次数
            self.total_commission += trade.commission  # 总的佣金

            stock_name = trade.data._name  # 股票名称
            self.log(
                f'股票:{stock_name},毛收益:{trade.pnl:.2f},扣佣后收益:{trade.pnlcomm:.2f},佣金:{trade.commission:.2f},总的佣金:{self.total_commission:.2f},资产市值:{self.broker.getvalue():.2f},'
                f'持仓市值:{self.get_position_value():.2f},现金:{self.broker.getcash():.2f},总收益:{self.total_profit:.2f},胜率:{100 * self.winning_trades / self.total_trades:.2f},盈亏比:{risk_reward_ratio:.2f},'
                f'开仓时间：{trade.dtopen},平仓时间：{trade.dtclose},交易头寸：{trade.size}')


    def notify_cashvalue(self, cash, value):
        """
        在每个交易日结束时打印当天日期、现金、资产市值和持仓市值，并在指定日期范围内记录资产市值

        参数:
            cash: 当前的现金余额
            value: 当前的资产市值
        """
        cur_date = self.data0.datetime.date(0).strftime('%Y-%m-%d')
        self.log(f'日期:{cur_date},现金:{cash},资产市值:{value},持仓市值:{self.get_position_value()}')
        if self.start_date <= cur_date <= self.end_date:
            self.value.append(value)  # 记录资产市值
            self.cash_history.append(cash)
            self.position_value_history.append(self.get_position_value())
            self.dates.append(cur_date)



    def stop(self):
        """
        策略结束时执行，计算最终结果并保存状态
        """
        # 使用基准计算器计算策略表现
        self.log(f'len(self.value){len(self.value)}')
        self.benchmark_portfolio_daily = self.benchmark_calculator.calculate_strategy_performance(self.value)
        # 打印回测日收益和基准日收益、收益差值数据
        self.log(f'回测日收益和基准日收益、收益差值数据:{self.benchmark_portfolio_daily}')

        self.log('(策略结束)')

        # 打印所有交易明细
        self.print_trade_details()

        # 计算交易统计信息
        if self.trade_stats:
            wins = [t for t in self.trade_stats if t['is_win']]
            losses = [t for t in self.trade_stats if not t['is_win']]

            total_trades = len(self.trade_stats)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0

            avg_win = np.mean([t['profit_loss_ratio'] for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t['profit_loss_ratio']) for t in losses]) if losses else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

            #计算金额统计
            max_win_amount=max([t['profit_loss_amount'] for t in wins]) if wins else 0
            max_loss_amount = min([t['profit_loss_amount'] for t in losses]) if losses else 0
            avg_win_amount = np.mean([t['profit_loss_amount'] for t in wins]) if wins else 0
            avg_loss_amount = np.mean([t['profit_loss_amount'] for t in losses]) if losses else 0

            print('=' * 80)
            print('交易统计:')
            print(f'总交易次数: {total_trades}')
            print(f'盈利次数: {len(wins)}')
            print(f'亏损次数: {len(losses)}')
            print(f'胜率: {win_rate:.2%}')
            print(f'平均盈利率: {avg_win:.2%}')
            print(f'平均亏损率: {avg_loss:.2%}')
            print(f'盈亏比: {profit_factor:.2f}')

            print('-'*40)
            print('金额统计(不含佣金)：')
            print(f"最大盈利金额：{max_win_amount:.2f}")
            print(f"最大亏损金额：{max_loss_amount:.2f}")
            print(f"平均盈利金额：{avg_win_amount:.2f}")
            print(f"平均亏损金额：{avg_loss_amount:.2f}")
            print('=' * 80)

            # 按卖出原因分类统计
            take_profit_count = len([t for t in self.trade_stats if t['sell_reason'] == 'take_profit'])
            stop_loss_count = len([t for t in self.trade_stats if t['sell_reason'] == 'stop_loss'])
            holding_period_count = len([t for t in self.trade_stats if t['sell_reason'] == 'holding_period'])
            print(f'其中止盈次数: {take_profit_count}')
            print(f'止损次数: {stop_loss_count}')
            print(f'持仓到期次数: {holding_period_count}')
            print('=' * 80)

            # 绘制盈亏比分布图
            self.plot_trade_statistics()

            #绘制资产变化图
            self.plot_asset_evolution()

            self.plot_pnl_distribution()



    def print_trade_details(self):
        """打印所有交易明细"""
        if not self.trade_stats:
            print("没有交易记录")
            return

        print("\n" + "="*100)
        print("交易明细汇总:")
        print("="*100)
        print(f"{'序号':<5} {'股票代码':<10} {'买入日期':<12} {'卖出日期':<12} {'买入价格':<10} {'卖出价格':<10} "
              f"{'收益率':<10} {'盈亏金额':<10} {'是否盈利':<8} {'卖出原因':<12}")
        print("-"*100)

        for i, trade in enumerate(self.trade_stats, 1):
            buy_date = trade['buy_date'].strftime('%Y-%m-%d') if hasattr(trade['buy_date'], 'strftime') else str(trade['buy_date'])
            sell_date = trade['sell_date'].strftime('%Y-%m-%d') if hasattr(trade['sell_date'], 'strftime') else str(trade['sell_date'])
            is_win = "是" if trade['is_win'] else "否"
            profit_color = '\033[92m' if trade['is_win'] else '\033[91m'  # 绿色表示盈利，红色表示亏损
            color_end = '\033[0m'
            amount=trade.get('profit_loss_amount',0)

            print(f"{i:<5} {trade['stock']:<10} {buy_date:<12} {sell_date:<12} "
                  f"{trade['buy_price']:<10.3f} {trade['sell_price']:<10.3f} "
                  f"{profit_color}{trade['profit_loss_ratio']:<10.2%}{color_end}{amount:<10.2f} {is_win:<8} {trade['sell_reason']:<12}")

        print("="*100)

        # 计算汇总统计
        total_return = sum(t['profit_loss_ratio'] for t in self.trade_stats)
        winning_trades = [t for t in self.trade_stats if t['is_win']]
        losing_trades = [t for t in self.trade_stats if not t['is_win']]

        if winning_trades:
            avg_win_return = np.mean([t['profit_loss_ratio'] for t in winning_trades])
            max_win = max([t['profit_loss_ratio'] for t in winning_trades])
        else:
            avg_win_return = 0
            max_win = 0

        if losing_trades:
            avg_loss_return = np.mean([t['profit_loss_ratio'] for t in losing_trades])
            max_loss = min([t['profit_loss_ratio'] for t in losing_trades])
        else:
            avg_loss_return = 0
            max_loss = 0

        print(f"\n汇总统计:")
        print(f"  总收益率: {total_return:.2%}")
        print(f"  平均每笔收益: {total_return/len(self.trade_stats) if self.trade_stats else 0:.2%}")
        print(f"  平均盈利: {avg_win_return:.2%}")
        print(f"  平均亏损: {avg_loss_return:.2%}")
        print(f"  最大盈利: {max_win:.2%}")
        print(f"  最大亏损: {max_loss:.2%}")


    def plot_trade_statistics(self):
        if not self.trade_stats:
            print("没有交易记录可供绘图")
            return

        # 提取盈亏比数据
        pl_ratios = [t['profit_loss_ratio'] for t in self.trade_stats]

        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 盈亏比分布直方图
        n, bins, patches = ax1.hist(pl_ratios, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('盈亏比')
        ax1.set_ylabel('频次')
        ax1.set_title('盈亏比分布')
        ax1.grid(True, alpha=0.3)

        # 累计收益曲线
        cumulative_returns = np.cumsum(pl_ratios)
        ax2.plot(cumulative_returns, marker='o', markersize=3)
        ax2.set_xlabel('交易次数')
        ax2.set_ylabel('累计收益率')
        ax2.set_title('累计收益曲线')
        ax2.grid(True, alpha=0.3)

        # 胜率和盈亏比可视化
        wins = [t for t in self.trade_stats if t['is_win']]
        losses = [t for t in self.trade_stats if not t['is_win']]

        win_rate = len(wins) / len(self.trade_stats) * 100 if self.trade_stats else 0

        if losses:
            avg_win = np.mean([t['profit_loss_ratio'] for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t['profit_loss_ratio']) for t in losses]) if losses else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_factor = 0

        x_pos = np.arange(1)
        width = 0.35

        bars1 = ax3.bar(x_pos - width/2, [win_rate], width, label='胜率(%)', alpha=0.7, color='green')
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x_pos + width/2, [profit_factor], width, label='盈亏比', alpha=0.7, color='orange')

        ax3.set_ylabel('胜率 (%)', color='green')
        ax3_twin.set_ylabel('盈亏比', color='orange')
        ax3.set_title('交易绩效指标')
        ax3.set_xticks([])

        # 添加数值标签
        ax3.bar_label(bars1, labels=[f'{win_rate:.1f}%'], padding=3)
        if profit_factor > 0:
            ax3_twin.bar_label(bars2, labels=[f'{profit_factor:.2f}'], padding=3)

        plt.tight_layout()
        plt.savefig('/home/quant/zc/backtrader/QuantBacktester_57/output.png',dpi=300,bbox_inches='tight')
        # plt.show()

    def plot_asset_evolution(self):
        """
        绘制资产变化图：包含现金、持仓市值和总资产
        """
        if not self.dates:
            print("没有资产数据可供绘图")
            return

        plt.figure(figsize=(15, 8))

        # 转换日期格式以便绘图
        dates = pd.to_datetime(self.dates)

        # 绘制三条线
        plt.plot(dates, self.value, label='Total Asset Value', color='blue', linewidth=2)
        plt.plot(dates, self.cash_history, label='Cash', color='green', linewidth=1.5, linestyle='--')
        plt.plot(dates, self.position_value_history, label='Position Value', color='red', linewidth=1.5, linestyle='-.')

        # 设置图表属性
        plt.title('Asset Evolution: Cash vs Position vs Total Value', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)

        # 格式化x轴日期显示
        plt.gcf().autofmt_xdate()

        # 保存图片
        save_path = '/home/quant/zc/backtrader/QuantBacktester_57/asset_evolution.png'
        # 也可以使用配置中的路径，这里为了简单起见使用了硬编码路径，建议改为参数控制
        # save_path = self.params.plot_path if hasattr(self.params, 'plot_path') else 'asset_evolution.png'

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"资产变化图已保存至: {save_path}")
        except Exception as e:
            print(f"保存资产变化图失败: {e}")

        # plt.show()

    def plot_pnl_distribution(self):
        """
        绘制盈亏金额分布图
        """
        if not self.trade_stats:
            print("没有交易记录可供绘图")
            return

        pnl_amounts = [t.get('profit_loss_amount', 0.0) for t in self.trade_stats]
        if not pnl_amounts:
            return

        plt.figure(figsize=(15, 8))

        # 创建颜色列表：大于0为红色，小于等于0为绿色
        colors = ['red' if x > 0 else 'green' for x in pnl_amounts]

        # 绘制柱状图
        x = range(len(pnl_amounts))
        plt.bar(x, pnl_amounts, color=colors, alpha=0.7)

        # 添加零轴线
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # 设置图表属性
        plt.title('Trade Profit/Loss Distribution (Amount)', fontsize=16)
        plt.xlabel('Trade ID', fontsize=12)
        plt.ylabel('Profit/Loss Amount', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')  # 只显示Y轴网格

        # 添加统计信息作为图例/注释
        max_win = max(pnl_amounts)
        max_loss = min(pnl_amounts)
        avg_pnl = np.mean(pnl_amounts)

        info_text = f'Max Win: {max_win:.2f}\nMax Loss: {max_loss:.2f}\nAvg PnL: {avg_pnl:.2f}'
        plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

        # 保存图片
        save_path = '/home/quant/zc/backtrader/QuantBacktester_57/pnl_distribution.png'

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"盈亏金额分布图已保存至: {save_path}")
        except Exception as e:
            print(f"保存盈亏金额分布图失败: {e}")

        # plt.show()