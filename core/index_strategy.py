import math

import backtrader as bt
import pandas as pd

from utils.config import config
from utils.helpers import cvxopt, get_stock_dic, get_st_stock,get_index_dic

# 导入新创建的组件
from core.trade_executor import TradeExecutor
from core.benchmark_calculator import BenchmarkCalculator
from core.position import Portfolio

class IndexStrategy(bt.Strategy):
    """
    IndexStrategy选股策略

    基于调仓表选择指数并进行交易
    """

    params = (
        ('period', config.backtest.strategy.period),  # 默认周期
        ('execution_price', config.backtest.strategy.execution_price),  # 默认为第二天开盘价
        ('max_position', config.backtest.strategy.max_position),  # 占资产总最大持仓比例
        ('start_date', config.backtest.strategy.start_date),  # 起始日期
        ('end_date', config.backtest.strategy.end_date),  # 结束日期
    )

    def __init__(self,BenchmarkDetailData,startDate, endDate, cash, commission, perc, rebalancing_days,weights=None,
                 dict_weights_raw=None,rebalance_plan=None, equal_weight=False):
        """
            # 初始化函数
            # 参数：
            # BenchmarkDetailData: 基准指数数据，包括['TRADEDATE', 'TCLOSE']，日期和指数收盘数值
            # startDate 起始日期
            # endDate 结束日期
            # codes 股票列表
            # cash 起始现金金额
            # rebalance_plan 调仓计划，格式为{日期：[股票代码列表]}
            # equal_weight 是否使用等权重

        """
        self.addminperiod(self.params.period)  # 最小周期数
        self.params.start_date = startDate  # 起始日期
        self.params.end_date = endDate  # 结束日期

        self.cash = cash  # 起始现金金额
        self.commission = commission  # 手续费比例
        self.perc = perc  # 滑点比例

        self.order = None

        self.order_list = []  # 记录以往订单，方便调仓日对未完成订单做处理

        self.FIFO_positions = {}  # 记录股票买入日期和size

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
        self.processed_dates = set()  # 记录已处理日期

        self.total_commission = 0  # 总的佣金


        # benchmark数据
        self.BenchmarkDetailData = BenchmarkDetailData

        self.monthly_rebalance_days = rebalancing_days

        self.weights = weights
        self.dict_weights_raw = dict_weights_raw

        # 初始化组件
        self.benchmark_calculator = BenchmarkCalculator(
            BenchmarkDetailData, self.params.start_date, self.params.end_date
        )
        self.benchmark_portfolio_daily = self.benchmark_calculator.get_benchmark_portfolio_daily()
        self.benchmark_annualReturn = self.benchmark_calculator.get_benchmark_annual_return()

        self.trade_executor = TradeExecutor(self, self.params)

        # 新增参数
        self.rebalance_plan = rebalance_plan or {}  # 调仓计划
        self.equal_weight = equal_weight  # 是否使用等权重

    def log(self, txt, dt=None):
        """
        日志函数

        参数:
            txt: 日志文本
            dt: 日期时间
        """
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def get_position_value(self):
        """
        计算当前所有持仓股票的总价值

        返回:
            当前所有持仓股票的总价值
        """
        position_value = 0
        for stock in self.FIFO_positions:
            stock_data = self.getdatabyname(stock)
            position_value += self.getposition(stock_data).size * stock_data.close[0]
        return position_value

    def prenext(self):
        """在策略正式开始前的预处理阶段调用"""
        self.next()

    def next(self):
        """
        主要策略逻辑
        每个bar都会调用这个方法
        """
        cur_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')

        # 调仓日处理
        if cur_date in self.monthly_rebalance_days:
            # 在调仓之前，取消之前所下的没成交也未到期的订单
            self.trade_executor.cancel_pending_orders()

            # 根据不同的策略调仓
            if self.rebalance_plan and cur_date in self.rebalance_plan:
                # 基于调仓表的等权重调仓
                weights, dict_weights_raw = self._rebalance_from_plan(cur_date)
                self.trade_executor.index_rebalance_portfolio(cur_date, weights, dict_weights_raw)


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
                # 记录头寸到FIFO_positions 先进先出的仓位数据中
                if stock not in self.FIFO_positions:
                    self.FIFO_positions[stock] = {}
                if str_buy_date not in self.FIFO_positions[stock]:
                    self.FIFO_positions[stock][str_buy_date] = 0
                # 在先进先出持仓字典中记录建仓时间和数量
                self.FIFO_positions[stock][str_buy_date] += order.executed.size
                self.log(
                    f'买单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
            elif order.issell():
                # 卖出指数，先进先出调整头寸
                index = order.data._name
                self.FIFO_Deal(index, order.executed.size)
                self.log(
                    f'卖单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
        else:
            self.log(
                f'订单作废  {order.data._name}, {order.getstatusname()}, isbuy:{order.isbuy()}, {order.created.size}, 创建时间 {bt.num2date(order.created.dt)}')

    def FIFO_Deal(self, index, size):
        """
        先进先出卖出指数
        指数卖出时，调整FIFO结构中指数持仓

        参数:
            index: 指数代码
            size: 卖出数量
        """
        if size < 0:
            size = 0 - size

        # 检查总持仓是否足够卖出
        total_size = sum(self.FIFO_positions.get(index, {}).values())
        if total_size < size:
            print(f'Error:指数{index}卖出数量大于总持仓({size}>{total_size})')
            return 0

        remaining_size = size  # 记录需要卖出的指数数量

        for buy_date in sorted(self.FIFO_positions[index].keys()):
            position_size = self.FIFO_positions[index][buy_date]
            if position_size > 0:
                if position_size > remaining_size:
                    # 更新持仓数量
                    self.FIFO_positions[index][buy_date] -= remaining_size
                    break
                else:
                    # 当前指数持仓数量小于剩余卖出数量，则扣减当前指数持仓数量，并继续循环
                    remaining_size -= position_size
                    # 删除已清空处理的持仓记录
                    del self.FIFO_positions[index][buy_date]

        # 删除已清空的指数记录
        if index in self.FIFO_positions and not self.FIFO_positions[index]:
            del self.FIFO_positions[index]


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

            index_name = trade.data._name  # 股票名称
            print(
                f'指数:{index_name},毛收益:{trade.pnl:.2f},扣佣后收益:{trade.pnlcomm:.2f},佣金:{trade.commission:.2f},总的佣金:{self.total_commission:.2f},资产市值:{self.broker.getvalue():.2f},'
                f'持仓市值:{self.get_position_value():.2f},现金:{self.broker.getcash():.2f},总收益:{self.total_profit:.2f},胜率:{100 * self.winning_trades / self.total_trades:.2f},盈亏比:{risk_reward_ratio:.2f}')


    def notify_cashvalue(self, cash, value):
        """
        在每个交易日结束时打印当天日期、现金、资产市值和持仓市值，并在指定日期范围内记录资产市值

        参数:
            cash: 当前的现金余额
            value: 当前的资产市值
        """
        cur_date = self.data0.datetime.date(0).strftime('%Y-%m-%d')

        # 如果当前日期未处理过，则记录
        if cur_date not in self.processed_dates:
            self.processed_dates.add(cur_date)  # 标记未已处理

            print(f'日期:{cur_date},现金:{cash},资产市值:{value},持仓市值:{self.get_position_value()}')
            if self.params.start_date <= cur_date <= self.params.end_date:
                self.value.append(value)  # 记录资产市值


    def stop(self):
        """
        策略结束时执行，计算最终结果并保存状态
        """
        # 使用基准计算器计算策略表现
        if self.value:
            self.benchmark_portfolio_daily = self.benchmark_calculator.calculate_strategy_performance(self.value)
            # 打印回测日收益和基准日收益、收益差值数据
            print(f'回测日收益和基准日收益、收益差值数据:{self.benchmark_portfolio_daily}')



    def _rebalance_from_plan(self, cur_date):
        """
        根据调仓计划进行等权重调仓

        参数：
            cur_date:当前日期
            cur_st_list:当前ST股票列表
        """
        print(f'根据调仓计划在{cur_date}进行等权重调仓')

        # 获取当前日期的目标指数权重字典
        weights = self.rebalance_plan.get(cur_date, {})
        # 转换为DataFrame
        df_weights = pd.DataFrame.from_dict(
            weights,
            orient='index',
            columns=['weight']
        ).reset_index()
        # 重命名列
        df_weights.columns = ['indexcode', 'weight']
        print(f'调仓日{cur_date}的目标指数数量：{len(df_weights)}')

        index_dic = get_index_dic(self.datas)
        current_value = self.broker.getvalue()  # 总资产价值

        # 持仓指数权重
        dict_weights_raw = {}
        # 持仓指数权重
        current_weights_raw = []
        # 持仓指数代码
        selected_indices = []
        for indexcode, buy_dates in self.FIFO_positions.items():
            selected_indices.append(indexcode)
            bs_size = sum(buy_dates.values())
            bs_price = index_dic[indexcode]['price']
            w = (bs_price * bs_size) / (
                    current_value * self.params.max_position)  # 价格*size/(current_value*self.params.max_position)
            w = round(w, 8)
            current_weights_raw.append(w)
            dict_weights_raw[indexcode] = w
        print('selected_indices:', selected_indices)

        return df_weights, dict_weights_raw
