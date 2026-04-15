
import backtrader as bt
import pandas as pd

from utils.config import config
from utils.helpers import cvxopt, get_stock_dic, get_st_stock,get_index_dic

# 导入新创建的组件
from core.dividend_handler import DividendHandler
from core.trade_executor import TradeExecutor
from core.state_manager import StateManager
from core.benchmark_calculator import BenchmarkCalculator
from core.position import Portfolio
from core.constraint_manager import create_constraint_manager_from_params


class AlphaStrategy(bt.Strategy):
    """
    Alpha选股策略
    
    基于预测分数(alpha)选择股票并进行交易
    """

    params = ()
    def __init__(self, st_dict, dividends, dividends_probonus, dividends_changert, BenchmarkDetailData, BaseStockDetailData,
                 startDate, endDate, cash, commission, perc, rebalancing_days, codeslist, weights=None, dict_weights_raw=None,
                 prev_date=None, is_first_run=True, saved_state=None,rebalance_plan=None):
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
            # rebalance_plan 调仓计划，格式为{日期：[股票代码列表]}

        """
        # 动态设置参数
        self.params = type('Params', (), {
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
            'save_state': config.backtest.strategy.save_state,  # 是否存储状态，开启或关闭增量回测
            'month_ibt': config.backtest.strategy.month_ibt,  # 是否每月增量回测
            'min_holding_days': config.backtest.strategy.min_holding_days,  # 最小持有天数,设置为0表示关闭持有时间约束
            'stock_pool_mode': config.backtest.strategy.stock_pool_mode,  # 股票池模式：0=关闭约束，1=启用约束
            'market_cap_threshold': config.backtest.strategy.market_cap_threshold,  # 市值阈值(亿元),设置为0表示关闭市值检查
            'consecutive_limit_days': config.backtest.strategy.consecutive_limit_days,  # 连续涨跌停天数限制,设置为0表示关闭约束
            'st_stock_mode': config.backtest.strategy.st_stock_mode,  # ST股票模式：0=关闭约束，1=启用约束
            'round_to_hundred': config.backtest.strategy.round_to_hundred,  # 是否将股票数量设置为100的整数倍
        })()

        self.addminperiod(self.params.period)  # 最小周期数
        self.start_date = startDate  # 起始日期
        self.end_date = endDate  # 结束日期

        self.cash = cash  # 起始现金金额
        self.commission = commission  # 手续费比例
        self.perc = perc  # 滑点比例

        self.order_list = []  # 记录以往订单，方便调仓日对未完成订单做处理

        # 统一的持仓管理器
        self.portfolio = Portfolio()
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
        self.total_commission = 0  # 总的佣金
        self.dates = []
        
        # st数据
        self.st_dict = st_dict

        # benchmark数据
        self.BenchmarkDetailData = BenchmarkDetailData
        # benchmark stock数据
        self.BaseStockDetailData = BaseStockDetailData

        # 增量回测
        self.to_restore = saved_state
        self.is_first_run = is_first_run
        self.prev_date = prev_date
        self.dividend_tax = {}  # 分红扣税记录

        self.monthly_rebalance_days = rebalancing_days

        # cvxopt
        self.codeslist = codeslist
        self.weights = weights
        self.dict_weights_raw = dict_weights_raw
        
        # 初始化组件
        self.benchmark_calculator = BenchmarkCalculator(
            BenchmarkDetailData, startDate, endDate
        )
        self.benchmark_portfolio_daily = self.benchmark_calculator.get_benchmark_portfolio_daily()
        self.benchmark_annualReturn = self.benchmark_calculator.get_benchmark_annual_return()
        
        self.dividend_handler = DividendHandler(
            self, dividends, dividends_probonus, dividends_changert, 
            self.portfolio, self.params
        )
        
        self.trade_executor = TradeExecutor(self, self.params)
        self.state_manager = StateManager(self, self.params)

        #新增参数
        self.rebalance_plan=rebalance_plan or {} #调仓计划

        # 将股票池列表转换为集合
        self.stock_pool = set(codeslist) if codeslist else set()
        market_cap_constraint_status = '禁用' if self.params.market_cap_threshold <= 0 else f"启用"
        self.log(f"市值检查：{market_cap_constraint_status}")
        # 初始化约束管理器
        self.constraint_manager = create_constraint_manager_from_params(self.params, self.stock_pool)
        # 约束状态统一显示
        constraint_status = self.constraint_manager.get_constraint_status()
        self.log(f"约束配置：{constraint_status}")
    
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
        for stock in self.portfolio.get_holding_stocks():
            stock_data = self.getdatabyname(stock)
            position_value += self.getposition(stock_data).size * stock_data.close[0]
        return position_value

    def build_constraint_context(self,st_list,current_date):
        return {
            'strategy':self,
            'st_list':st_list,
            'current_date':current_date,
        }

    def enforce_min_holding_weight_floor(self,weights_df,dict_weights_raw,cur_st_list,current_date):
        if self.params.min_holding_days<=0:
            return weights_df
        context=self.build_constraint_context(cur_st_list,current_date)
        adjusted=weights_df.copy()
        adjusted=adjusted.groupby('code',as_index=False)['weight'].sum()
        locked_count=0
        for stock,current_weight in dict_weights_raw.items():
            passed,_=self.constraint_manager.check_single_constraint(stock,context,'MinHoldingDays')
            if passed:
                continue
            locked_count+=1
            if stock in adjusted['code'].values:
                idx=adjusted.index[adjusted['code']==stock][0]
                if adjusted.at[idx,'weight']<current_weight:
                    adjusted.at[idx,'weight']=current_weight
            else:
                adjusted=pd.concat([adjusted,pd.DataFrame([{'code':stock,'weight':current_weight}])],ignore_index=True)
        if locked_count>0:
            self.log(f"最小持有期保护股票数量：{locked_count}")
        return adjusted


    
    def prenext(self):
        """在策略正式开始前的预处理阶段调用"""
        self.next()
    
    def next(self):
        """
        主要策略逻辑
        每个bar都会调用这个方法
        """
        cur_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')

        # 增量回测恢复状态
        if self.state_manager.restore_state(self.prev_date, self.to_restore, self.portfolio, self.dividend_tax, self.is_first_run):
            return
        
        # 找到截至当前日期的最近日期的ST股票列表
        cur_st_list = get_st_stock(self.st_dict, cur_date)
        
        # 卖出st持仓股票
        self.trade_executor.sell_st_stocks(cur_st_list)
        
        # 分红处理
        # len(self.data)为已处理数据的长度,self.data._idx为数据总长度,最后一天不处理分红
        if len(self.data) - self.data._idx != 0:
            self.dividend_handler.deal_dividend(cur_date)

        # 调仓日处理
        if cur_date in self.monthly_rebalance_days:
            # 在调仓之前，取消之前所下的没成交也未到期的订单
            self.trade_executor.cancel_pending_orders()
            
            # 根据不同的策略调仓
            if self.rebalance_plan and cur_date in self.rebalance_plan:
                #基于调仓表的等权重调仓
                weights,dict_weights_raw=self._rebalance_from_plan(cur_date)
                self.trade_executor.rebalance_portfolio(cur_date, weights, dict_weights_raw)
            elif self.params.month_ibt:
                weights = self.weights
                dict_weights_raw = self.dict_weights_raw
                self.trade_executor.rebalance_portfolio(cur_date, weights, dict_weights_raw)
            else:
                stock_dic = get_stock_dic(self.datas)
                current_value = self.broker.getvalue()  # 总资产价值
                #将Portfolio对象转换为cvxopt函数
                fifo_positions_dict={}
                for stock in self.portfolio.get_holding_stocks():
                    positions=self.portfolio.get_stock_positions(stock)
                    fifo_positions_dict[stock]={pos.buy_date:pos.size for pos in positions.values()}

                weights, dict_weights_raw = cvxopt(cur_date, cur_st_list, self.codeslist, self.BaseStockDetailData,
                                                  current_value, stock_dic, fifo_positions_dict,
                                                  self.params.max_position,self.constraint_manager,self)
                weights=self.enforce_min_holding_weight_floor(weights,dict_weights_raw,cur_st_list,cur_date)
                self.trade_executor.rebalance_portfolio(cur_date, weights, dict_weights_raw)


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
                # 使用Portfolio对象记录持仓
                self.portfolio.add_position(stock, str_buy_date, order.executed.size,order.executed.price)

                self.log(
                    f'买单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
            elif order.issell():
                # 卖出股票，扣税, 先进先出调整头寸
                stock = order.data._name
                current_date = bt.num2date(order.executed.dt)
                tax = self.dividend_handler.get_dividend_tax(stock, order.executed.size, current_date)

                # 使用Portfolio对象处理卖出（FIFO原则）
                self.portfolio.remove_position(stock, order.executed.size)

                if self.params.deal_dividend and tax != 0:
                    tax_date = current_date.strftime('%Y-%m-%d')
                    print(f'在{tax_date}股票{stock}分红扣税:{tax}')
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
            print(f'Order Size:{order.size},Price:{order.data.close[0]},Total Cost:{total_cost},Available Cash:{cash}')
            if total_cost > cash:
                print('Confirmed:Margin due to insufficient balance')


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
            print(f'股票:{stock_name},毛收益:{trade.pnl:.2f},扣佣后收益:{trade.pnlcomm:.2f},佣金:{trade.commission:.2f},总的佣金:{self.total_commission:.2f},资产市值:{self.broker.getvalue():.2f},'
                  f'持仓市值:{self.get_position_value():.2f},现金:{self.broker.getcash():.2f},总收益:{self.total_profit:.2f},胜率:{100 * self.winning_trades / self.total_trades:.2f},盈亏比:{risk_reward_ratio:.2f}')


    def notify_cashvalue(self, cash, value):
        """
        在每个交易日结束时打印当天日期、现金、资产市值和持仓市值，并在指定日期范围内记录资产市值
        
        参数:
            cash: 当前的现金余额
            value: 当前的资产市值
        """
        cur_date = self.data0.datetime.date(0).strftime('%Y-%m-%d')
        print(f'日期:{cur_date},现金:{cash},资产市值:{value},持仓市值:{self.get_position_value()}')
        if self.start_date <= cur_date <= self.end_date:
            self.dates.append(cur_date)
            self.value.append(value)  # 记录资产市值

    
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
        
        cur_date = self.data0.datetime.date(0).strftime('%Y-%m-%d')
        
        # 保存状态
        self.state_manager.save_state(
            cur_date, self.benchmark_portfolio_daily, self.portfolio,
            self.dividend_tax, self.reward, self.rewardcnt, self.risk, self.riskcnt
        )


    def _rebalance_from_plan(self,cur_date):
        """
        根据调仓计划进行等权重调仓

        参数：
            cur_date:当前日期
            cur_st_list:当前ST股票列表
        """
        print(f'根据调仓计划在{cur_date}进行等权重调仓')

        #获取当前日期的目标股票权重字典
        weights=self.rebalance_plan.get(cur_date, {})
        #转换为DataFrame
        df_weights=pd.DataFrame.from_dict(
            weights,
            orient='index',
            columns=['weight']
        ).reset_index()
        #重命名列
        df_weights.columns=['code','weight']
        print(f'调仓日{cur_date}的目标股票数量：{len(df_weights)}')

        stock_dic = get_stock_dic(self.datas)
        current_value = self.broker.getvalue()  # 总资产价值

        # 持仓股票权重
        dict_weights_raw = {}
        # 持仓股票代码
        selected_indices = []
        for stock in self.portfolio.get_holding_stocks():
            selected_indices.append(stock)
            bs_size = self.portfolio.get_stock_total_size(stock)
            bs_price = stock_dic[stock]['price']
            w = (bs_price * bs_size) / (
                    current_value * self.params.max_position)  # 价格*size/(current_value*self.params.max_position)
            w = round(w, 8)
            dict_weights_raw[stock] = w
        print('selected_indices:', selected_indices)

        return df_weights,dict_weights_raw

