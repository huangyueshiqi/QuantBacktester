import json

import math
import os
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

from utils.config import config
from utils.helpers import cvxopt, get_stock_dic, get_st_stock

# 导入新创建的组件
from core.dividend_handler import DividendHandler
from core.trade_executor import TradeExecutor
from core.state_manager import StateManager
from core.benchmark_calculator import BenchmarkCalculator
from core.position import Portfolio
from core.constraint_manager import create_constraint_manager_from_params



class LLMFilterParser:
    field_aliases={
        'roe':'ROE',
        '净资产收益率':'ROE',
        'pb':'PB',
        '市净率':'PB'
    }
    supported_operators={'>','>=','<','<=','==','!='}

    @classmethod
    def parse_structured_filter_json(cls,json_text_or_path):
        if not json_text_or_path:
            return {'logic':'and','conditions':[]}
        if os.path.exists(json_text_or_path):
            with open(json_text_or_path,'r',encoding='utf-8') as f:
                content=f.read()
        else:
            content=json_text_or_path
        data=json.loads(content)
        if isinstance(data,dict) and 'conditions' in data:
            normalized={'logic':str(data.get('logic','and')).lower(),'conditions':[]}
            for item in data.get('conditions',[]):
                field=str(item.get('field','')).strip()
                if not field:
                    continue
                field_key=field.lower()
                mapped_field=cls.field_aliases.get(field_key,field.upper())
                operator=str(item.get('operator','>')).strip()
                if operator not in cls.supported_operators:
                    continue
                value=float(item.get('value',0))
                normalized['conditions'].append({
                    'field':mapped_field,
                    'operator':operator,
                    'value':value
                })
            return normalized
        if isinstance(data,list):
            return cls.parse_structured_filter_json(json.dumps({'logic':'and','conditions':data}))
        return {'logic':'and','conditions':[]}

    @staticmethod
    def apply_structured_filter(dataframe,structured_filter):
        if dataframe is None or dataframe.empty:
            return dataframe
        conditions=(structured_filter or {}).get('conditions',[])
        if not conditions:
            return dataframe
        operator_map={
            '>': lambda s, v: s > v,
            '>=': lambda s, v: s >= v,
            '<': lambda s, v: s < v,
            '<=': lambda s, v: s <= v,
            '==': lambda s, v: s == v,
            '!=': lambda s, v: s != v,
        }
        normalized_columns={str(c).strip().upper():c for c in dataframe.columns}
        masks=[]
        for condition in conditions:
            field=str(condition.get('field','')).strip().upper()
            if field not in normalized_columns:
                raise ValueError(f"筛选字段不存在：{field}")
            origin_col=normalized_columns[field]
            series=pd.to_numeric(dataframe[origin_col],errors='coerce')
            comparator=operator_map[condition['operator']]
            masks.append(comparator(series,float(condition['value'])))
        logic=str((structured_filter or {}).get('logic','and')).lower()
        combined=masks[0]
        for mask in masks[1:]:
            combined=(combined&mask) if logic!='or' else (combined|mask)
        return dataframe[combined.fillna(False)]

class LLMStrategy(bt.Strategy):
    """
    Buy选股策略
    """
    params=()

    def __init__(self, st_dict, dividends, dividends_probonus, dividends_changert, BenchmarkDetailData,
                 BaseStockDetailData, start_date,end_date,cash, commission, perc, codeslist,trade_list):
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
            'weighting_method': getattr(config.backtest.strategy, 'weighting_method', 'equal'),
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
        self.signal_data = trade_list
        self.signal_has_action=trade_list is not None and 'action' in trade_list.columns
        self.keep_holding_if_no_signal = self.signal_has_action
        self.holding_periods = 40  # 持仓天数
        self.take_profit = 0.5
        self.stop_loss = 0.3
        self.order_reasons = {}
        self.verbose=False

        # 添加交易统计相关变量
        self.trade_stats = []  # 存储交易统计数据
        self.trade_points={}

        # 资金管理变量
        self.total_days = 40  # 总份数
        self.accumulated_cash = 0  # 累积资金

        self.adjust_dates=set(pd.to_datetime(trade_list['datetime']).dt.strftime("%Y-%m-%d").unique())


        self.signal_dict=self.load_data(code_list=codeslist,trade_list=trade_list)
        self.orders={}
        self.pending_buy_signals=[]
        self.daily_stock_snapshots=[]
        self.stock_total_pnl_gross={}
        self.stock_trade_count={}
        self.single_stock_max_weight=0.4


    def load_data(self,code_list,trade_list):
        signal_dict={}
        for stock in code_list:
            stock_signals=trade_list[trade_list['instrument']==stock].copy()
            stock_signals['datetime'] = pd.to_datetime(stock_signals['datetime']).dt.strftime('%Y-%m-%d')
            stock_signals.set_index('datetime',inplace=True)
            signal_dict[stock]=stock_signals
        return signal_dict



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
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        if current_date not in self.adjust_dates:
            return

        # 1. 确定今日的目标持仓股票列表 (Target Stocks)
        target_stocks = []
        for data in self.datas:
            stock_name = data._name
            current_signal = self.get_signal_for_stock(stock_name, current_date)

            is_held = self.getposition(data).size > 0

            if current_signal is not None:
                default_action=1 if not self.signal_has_action else 0
                default_weight = 1 if not self.signal_has_action else 0
                action = int(float(current_signal.get('action',default_action)))
                weight = float(current_signal.get('weight', default_weight))
                if action == -1:
                    # 卖出信号，剔除出目标名单
                    continue
                elif action == 1 or weight > 0:
                    # 买入信号，加入目标名单
                    target_stocks.append(stock_name)
                    continue

            # 如果没有新信号，但当前持有，则继续保持在目标名单中
            if is_held and self.keep_holding_if_no_signal:
                target_stocks.append(stock_name)

        # 2. 调用通用的同日买卖、等权调仓逻辑
        self.adjust_target_portfolio(current_date, target_stocks)

    def adjust_target_portfolio(self, current_date, target_stocks):
        """
        通用目标持仓调仓逻辑（自动处理同日买卖，等权分配）
        """
        # 1. 撤销旧挂单
        self.cancel_pending_orders()

        # 2. 如果今天没有目标股票，直接清仓所有已有持仓
        if not target_stocks:
            for data in self.datas:
                if self.getposition(data).size > 0:
                    self.order_list.append(self.close(data=data))
            return

        weighting_method = str(getattr(self.params, 'weighting_method', 'equal')).lower()
        caps = {}
        if weighting_method == 'mktcap':
            for data in self.datas:
                stock_code = data._name
                if stock_code not in target_stocks:
                    continue
                cap_val = None
                if hasattr(data, 'MARKETVALUE'):
                    try:
                        cap_val = float(data.MARKETVALUE[0])
                    except Exception:
                        cap_val = None
                if cap_val is not None and cap_val > 0:
                    caps[stock_code] = cap_val
        use_mktcap = weighting_method == 'mktcap' and len(caps) == len(target_stocks) and sum(caps.values()) > 0
        if use_mktcap:
            raw_weights = {k: v / sum(caps.values()) for k, v in caps.items()}
            target_weights = {k: min(self.params.max_position * w, self.single_stock_max_weight) for k, w in raw_weights.items()}

            capped = {k for k, w in target_weights.items() if abs(w - self.single_stock_max_weight) < 1e-12}
            for _ in range(10):
                total_assigned = sum(target_weights.values())
                remaining = self.params.max_position - total_assigned
                if remaining <= 1e-12:
                    break
                free = [k for k in target_stocks if k not in capped]
                if not free:
                    break
                free_raw_sum = sum(raw_weights[k] for k in free)
                if free_raw_sum <= 0:
                    break
                updated = False
                for k in free:
                    add = remaining * (raw_weights[k] / free_raw_sum)
                    new_w = target_weights[k] + add
                    if new_w >= self.single_stock_max_weight:
                        target_weights[k] = self.single_stock_max_weight
                        capped.add(k)
                        updated = True
                    else:
                        target_weights[k] = new_w
                if not updated:
                    break
        else:
            target_weight = min(self.params.max_position / len(target_stocks), self.single_stock_max_weight)
            target_weights = {k: target_weight for k in target_stocks}
        current_value = self.broker.getvalue()

        operations = []  # 记录格式: (股票数据, 权重差值)

        # 4. 遍历所有股票，计算权重差值
        for data in self.datas:
            stock_code = data._name

            # 停牌检查
            if hasattr(data, 'tradestatus') and data.tradestatus[0] == 0:
                self.log(f"【{current_date}】股票 {stock_code} 停牌，跳过处理")
                continue

            current_size = self.getposition(data).size

            # 计算当前实际权重
            if current_value > 0:
                current_weight = (current_size * data.close[0]) / current_value
            else:
                current_weight = 0.0

            # 判断目标权重
            expected_weight = float(target_weights.get(stock_code, 0.0))

            # 计算权重差 (正数为买，负数为卖)
            weight_diff = expected_weight - current_weight

            # 设置一个极小的阈值(如0.005，即0.5%)，防止微调产生过多手续费
            if abs(weight_diff) > 0.005:
                operations.append((data, weight_diff, current_weight))

        # ================= 5. 核心：同一天买卖的执行顺序 =================

        # --- 第一步：先执行卖出 (weight_diff < 0) ---
        sell_operations = sorted([op for op in operations if op[1] < 0], key=lambda x: x[1])
        for data, weight_diff, current_weight in sell_operations:
            order=None
            # 跌停防守：如果接近跌停价，卖不出
            # if hasattr(data, 'stopping') and data.stopping[1] * self.params.low_stopping > data.low[1]:
            #     continue

            if weight_diff <= -current_weight:
                # 完全清仓
                order = self.close(data=data)
            else:
                # 部分减仓
                raw_sell_unit = (current_value * abs(weight_diff)) / data.close[0]
                sell_unit = math.floor(raw_sell_unit / 100) * 100 if self.params.round_to_hundred else math.floor(
                    raw_sell_unit)
                if sell_unit > 0:
                    order = self.sell(data=data, size=sell_unit)

            if  order:
                self.order_list.append(order)
                self.log(f"执行卖出/减仓 - {data._name}, 目标差额: {weight_diff:.2%}")

        # --- 第二步：再执行买入 (weight_diff > 0) ---
        buy_operations = sorted([op for op in operations if op[1] > 0], key=lambda x: x[1], reverse=True)
        for data, weight_diff, current_weight in buy_operations:
            # 涨停防守：如果接近涨停价，买不到
            # if hasattr(data, 'limit') and data.limit[1] * self.params.high_limit < data.high[1]:
            #     continue

            raw_buy_unit = (current_value * weight_diff) / data.close[0]
            buy_unit = math.floor(raw_buy_unit / 100) * 100 if self.params.round_to_hundred else math.floor(
                raw_buy_unit)

            if buy_unit > 0:
                order = self.buy(data=data, size=buy_unit)
                if order:
                    self.order_list.append(order)
                    self.log(f"执行买入/加仓 - {data._name}, 目标差额: {weight_diff:.2%}, 数量: {buy_unit}")

    def get_signal_for_stock(self, stock_name, current_date):
        """
        获取指定股票在当前日期的信号
        """
        if stock_name not in self.signal_dict:
            return None

        signal_df = self.signal_dict[stock_name]

        # 查找当前日期的信号
        mask = signal_df.index == current_date
        if not mask.any():
            return None

        return signal_df[mask].iloc[0]


    def execute_signal(self, data, signal,allocated_cash=None):
        """
        执行单个股票的交易信号
        """
        stock_name = data._name
        current_action = signal['action']
        current_weight = signal['weight']

        position = self.getposition(data)
        price = data.close[0]

        if current_action == 1:  # 买入信号
            if allocated_cash is None:
                total_value = self.broker.get_cash()
                allocated_cash=total_value * current_weight
            size = int(allocated_cash*0.95 / price)
            size=math.floor(size/100)*100
            order = self.buy(data=data, size=size)  # 稍微提高买入价以提高成交概率
            self.orders[stock_name] = order

            self.log(f'买入信号执行 - {stock_name}, 日期:{self.datas[0].datetime.date(0)}, '
                     f'价格:{price:.2f}, 数量:{size}, 权重:{current_weight:.2f}')

        elif current_action == -1:  # 卖出信号
            order = self.close(data=data)
            self.orders[stock_name] = order

            self.log(f'卖出信号执行 - {stock_name}, 日期:{self.datas[0].datetime.date(0)}, '
                     f'价格:{price:.2f}, 数量:{position.size}')


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
                self.trade_points.setdefault(stock, []).append({
                    'date': bt.num2date(order.executed.dt),
                    'price': float(order.executed.price),
                    'size': float(abs(order.executed.size)),
                    'side': 'buy'
                })

                #使用Portfolio对象记录持仓
                self.portfolio.add_position(stock,str_buy_date,order.executed.size,order.executed.price)

                self.log(
                    f'买单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
            elif order.issell():
                # 卖出股票，扣税, 先进先出调整头寸
                stock = order.data._name
                current_date = bt.num2date(order.executed.dt)
                self.trade_points.setdefault(stock, []).append({
                    'date': bt.num2date(order.executed.dt),
                    'price': float(order.executed.price),
                    'size': float(abs(order.executed.size)),
                    'side': 'sell'
                })

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

    def plot_trade_points(self):
        output_dir = os.path.abspath(getattr(config.paths, 'plot', 'plot'))
        os.makedirs(output_dir, exist_ok=True)

        for stock, points in self.trade_points.items():
            if not points:
                continue
            data = self.getdatabyname(stock)
            date_values = []
            close_values = []
            for dt, close in zip(data.datetime.array, data.close.array):
                if pd.isna(dt) or pd.isna(close):
                    continue
                date_values.append(bt.num2date(dt))
                close_values.append(float(close))
            if not date_values:
                continue

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(date_values, close_values, label='Close', linewidth=0.6, color='#1f77b4')

            buy_points = [point for point in points if point['side'] == 'buy']
            sell_points = [point for point in points if point['side'] == 'sell']

            if buy_points:
                ax.scatter(
                    [point['date'] for point in buy_points],
                    [point['price'] for point in buy_points],
                    marker='o',
                    color='red',
                    s=10,
                    label='Buy'
                )
            if sell_points:
                ax.scatter(
                    [point['date'] for point in sell_points],
                    [point['price'] for point in sell_points],
                    marker='o',
                    color='green',
                    s=10,
                    label='Sell'
                )

            ax.set_title(f'{stock} 买卖点')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            ax.legend()

            output_path = os.path.join(output_dir, f'llm_{stock}_trades.png')
            fig.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.log(f'买卖点图已保存: {output_path}')


    def plot_portfolio_stock_deep_dive(self):
        if not self.daily_stock_snapshots:
            return
        output_dir = os.path.abspath(getattr(config.paths, 'plot', 'plot'))
        os.makedirs(output_dir, exist_ok=True)
        snapshots = pd.DataFrame(self.daily_stock_snapshots)
        if snapshots.empty:
            return
        snapshots["date"] = pd.to_datetime(snapshots["date"])
        snapshots = snapshots.sort_values(["date", "stock"]).reset_index(drop=True)
        total_value_df = (
            snapshots[["date", "total_value"]]
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        total_value_df["prev_total_value"] = total_value_df["total_value"].shift(1)
        snapshots = snapshots.merge(
            total_value_df[["date", "prev_total_value"]], on="date", how="left"
        )
        snapshots["prev_market_value"] = snapshots.groupby("stock")["market_value"].shift(1)
        snapshots["prev_close"] = snapshots.groupby("stock")["close"].shift(1)
        valid_price_mask = snapshots["prev_close"] > 0
        snapshots["stock_return"] = np.where(
            valid_price_mask,
            snapshots["close"] / snapshots["prev_close"] - 1.0,
            0.0,
        )
        valid_contrib_mask = snapshots["prev_total_value"] > 0
        snapshots["contribution"] = np.where(
            valid_contrib_mask,
            (snapshots["prev_market_value"].fillna(0.0) / snapshots["prev_total_value"])
            * snapshots["stock_return"],
            0.0,
        )
        contrib_by_stock = snapshots.groupby("stock", as_index=True)["contribution"].sum().sort_values(ascending=False)
        if contrib_by_stock.empty:
            return

        stock_mv_full = (
            snapshots.pivot_table(index="date", columns="stock", values="market_value", aggfunc="sum")
            .fillna(0.0)
            .sort_index()
        )
        total_value_series = total_value_df.set_index("date")["total_value"]
        stock_weight_df = stock_mv_full.divide(total_value_series, axis=0).fillna(0.0)
        if not stock_weight_df.empty:
            top_stocks_weight = (
                stock_weight_df.mean().sort_values(ascending=False).head(9).index.tolist()
            )
            plot_weight_df = stock_weight_df[top_stocks_weight].copy() if top_stocks_weight else stock_weight_df.copy()
            remaining_weight_cols = [c for c in stock_weight_df.columns if c not in plot_weight_df.columns]
            if remaining_weight_cols:
                plot_weight_df["其他"] = stock_weight_df[remaining_weight_cols].sum(axis=1)
            fig, ax = plt.subplots(figsize=(14, 7))
            plot_weight_df.plot.area(ax=ax, linewidth=0, stacked=True)
            ax.set_title("组合内个股市值占比随时间变化 Top10")
            ax.set_xlabel("日期")
            ax.set_ylabel("占组合总资产比重")
            ax.set_ylim(0.0, 1.0)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            ax.legend(loc="upper left", ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "llm_stock_weight_stacked.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)



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
            self.stock_total_pnl_gross[stock_name]=self.stock_total_pnl_gross.get(stock_name,0)+trade.pnl
            self.stock_trade_count[stock_name]=self.stock_trade_count.get(stock_name,0)+1
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

            cur_dt=pd.to_datetime(cur_date)
            for data in self.datas:
                size=float(self.getposition(data).size)
                close_price=float(data.close[0])
                self.daily_stock_snapshots.append(
                    {
                        "date":cur_dt,
                        "stock":data._name,
                        "size":size,
                        "close":close_price,
                        "market_value":size*close_price,
                        "cash":float(cash),
                        "total_value":float(value)
                    }
                )



    def stop(self):
        """
        策略结束时执行，计算最终结果并保存状态
        """
        # 使用基准计算器计算策略表现
        self.log(f'len(self.value){len(self.value)}')
        benchmark_dates = list(self.benchmark_calculator.benchmark_portfolio_daily.index.astype(str))
        strategy_dates = [str(d) for d in self.dates]
        benchmark_only = sorted(list(set(benchmark_dates) - set(strategy_dates)))
        strategy_only = sorted(list(set(strategy_dates) - set(benchmark_dates)))
        self.log(f'策略日期数量:{len(strategy_dates)},基准日期数量:{len(benchmark_dates)}')
        self.log(f'仅基准日期数量:{len(benchmark_only)},仅策略日期数量:{len(strategy_only)}')
        if benchmark_only:
            self.log(f'仅基准日期样例:{benchmark_only[:5]}')
        if strategy_only:
            self.log(f'仅策略日期样例:{strategy_only[:5]}')

        self.benchmark_portfolio_daily = self.benchmark_calculator.calculate_strategy_performance(
            self.value, self.dates
        )
        # 打印回测日收益和基准日收益、收益差值数据
        self.log(f'回测日收益和基准日收益、收益差值数据:{self.benchmark_portfolio_daily}')
        self.plot_trade_points()

        self.plot_asset_evolution()
        self.plot_portfolio_stock_deep_dive()

        if self.stock_total_pnl_gross:
            sorted_pnl=sorted(self.stock_total_pnl_gross.items(),key=lambda x:x[1],reverse=True)
            for stock,gross_pnl in sorted_pnl:
                trade_count=self.stock_trade_count.get(stock,0)
                self.log(f"股票：{stock},总盈亏(毛)：{gross_pnl:.2f},完成交易数：{trade_count}")

        self.log('(策略结束)')


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

        # bench_copy=self.BenchmarkDetailData.copy()
        # bench_copy['TRADEDATE']=pd.to_datetime(bench_copy['TRADEDATE'])
        # need_bench=bench_copy[bench_copy['TRADEDATE'].isin(dates)]
        # benchdata=list(need_bench['TCLOSE'])
        # plt.plot(dates, benchdata, label='BenchmarkDetailData Value', color='orange', linewidth=2)
        # corr=np.corrcoef(self.value,benchdata)[0,1]
        # print(f"策略与基准相关系数：{corr:.3f}")
        #
        # from sklearn.linear_model import LinearRegression
        # benchdata=np.array(benchdata)
        # X = benchdata.reshape(-1, 1)
        # y = np.array(self.value)
        # model = LinearRegression().fit(X, y)
        # beta = model.coef_[0]
        # print(f"策略贝塔: {beta:.3f}")
        # r_squared = model.score(X, y)
        # print(f"R²: {r_squared:.3f}")

        # 设置图表属性
        plt.title('Asset Evolution: Cash vs Position vs Total Value', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)

        # 格式化x轴日期显示
        plt.gcf().autofmt_xdate()

        # 保存图片
        save_path = '/home/quant/zc/backtrader/QuantBacktester_57/plot/asset_evolution.png'
        # 也可以使用配置中的路径，这里为了简单起见使用了硬编码路径，建议改为参数控制
        # save_path = self.params.plot_path if hasattr(self.params, 'plot_path') else 'asset_evolution.png'

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"资产变化图已保存至: {save_path}")
        except Exception as e:
            print(f"保存资产变化图失败: {e}")










