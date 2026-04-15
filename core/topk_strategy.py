import math

import backtrader as bt
import pandas as pd

from utils.config import config
from utils.helpers import cvxopt, get_stock_dic, get_st_stock

# 导入新创建的组件
from core.dividend_handler import DividendHandler
from core.trade_executor import TradeExecutor
from core.state_manager import StateManager
from core.benchmark_calculator import BenchmarkCalculator
from core.position import Portfolio
from core.constraint_manager import create_constraint_manager_from_params


class TopkStrategy(bt.Strategy):
    """
    Topk选股策略

    基于调仓表实现topk策略：
    - 初始建仓50只股票
    - 在调仓日卖出仓内分数最低的5只股票
    - 买入不在仓内分数最高的5只股票
    -买入卖出采用等权分配，卖出全卖
    """
    params=()

    def __init__(self, st_dict, dividends, dividends_probonus, dividends_changert, BenchmarkDetailData,
                 BaseStockDetailData, start_date,end_date,cash, commission, perc, adjusting_days, codeslist, weights=None,
                 dict_weights_raw=None,filepath=None,prev_date=None, is_first_run=True, saved_state=None, adjust_data=None,rebalancing_days=None):
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
            'save_state': config.backtest.strategy.save_state,  # 是否存储状态，开启或关闭增量回测
            'month_ibt': config.backtest.strategy.month_ibt,  # 是否每月增量回测
            'initial_stocks': config.backtest.strategy.initial_stocks, #初始建仓股票数量
            'adjust_count': config.backtest.strategy.adjust_count,  # 每次调仓的股票数量
            'min_holding_days': config.backtest.strategy.min_holding_days,  # 最小持有天数,设置为0表示关闭持有时间约束
            'stock_pool_mode': config.backtest.strategy.stock_pool_mode,  # 股票池模式：0=关闭约束，1=启用约束
            'market_cap_threshold': config.backtest.strategy.market_cap_threshold,  # 市值阈值(亿元),设置为0表示关闭市值检查
            'consecutive_limit_days': config.backtest.strategy.consecutive_limit_days,  # 连续涨跌停天数限制,设置为0表示关闭约束
            'st_stock_mode': config.backtest.strategy.st_stock_mode,  # ST股票模式：0=关闭约束，1=启用约束
            'enable_rebalance': config.backtest.strategy.enable_rebalance,  # 是否启用再平衡功能，True启用，False禁用
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

        self.total_commission = 0  # 总的佣金

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

        self.monthly_adjust_days = adjusting_days
        self.monthly_rebalance_days = rebalancing_days

        # cvxopt
        self.weights = weights
        self.dict_weights_raw = dict_weights_raw

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

        # 新增参数
        self.adjust_data = adjust_data

        #持仓管理
        self.is_initialized=False  #是否已经初始化建仓
        #将股票池列表转换为集合
        self.stock_pool = set(codeslist) if codeslist else set()

        #调仓记录表
        self.adjust_records=[] #记录实际成功的交易操作
        self.filepath=filepath

        market_cap_constraint_status='禁用' if self.params.market_cap_threshold<=0 else f"启用"
        self.log(f"市值检查：{market_cap_constraint_status}")

        rebalance_status = '启用' if self.params.enable_rebalance else f"禁用"
        self.log(f"再平衡功能：{rebalance_status}")

        #初始化约束管理器
        self.constraint_manager=create_constraint_manager_from_params(self.params,self.stock_pool)

        #约束状态统一显示
        constraint_status=self.constraint_manager.get_constraint_status()
        self.log(f"Topk策略初始化完成:初始建仓{self.params.initial_stocks}只，每次调仓{self.params.adjust_count}只")
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

    def _build_constraint_context(self,st_list,current_date):
        """
        构建约束检查所需的上下文数据
        """
        return {
            'strategy':self,
            'st_list':st_list,
            'current_date':current_date,
        }

    def get_stock_market_cap(self,stock_code):
        """
        获取股票在指定日期的市值（亿元）
        """
        bs_data=self.getdatabyname(stock_code)

        #市值单位转换为亿元
        market_value=bs_data.MARKETVALUE[0]
        return market_value/100000000 if market_value else 0


    def get_market_cap_statistics(self,stock_list):
        """
        统计股票列表中大小市值股票的数量
        """
        if self.params.market_cap_threshold<=0:
            return len(stock_list),0

        large_cap_count=0
        small_cap_count = 0

        for stock in stock_list:
            market_cap=self.get_stock_market_cap(stock)
            if market_cap>=self.params.market_cap_threshold:
                large_cap_count+=1
            else:
                small_cap_count+=1

        return large_cap_count,small_cap_count


    def print_market_cap_statistics(self,title,stock_list):
        """
        打印股票列表的市值统计信息
        """
        if self.params.market_cap_threshold<=0:
            self.log(f"{title}:总计{len(stock_list)}只股票(未设置市值阈值)")
            return

        large_cap_count,small_cap_count=self.get_market_cap_statistics(stock_list)
        self.log(f"{title}:总计{len(stock_list)}只股票,大市值{large_cap_count}只，小市值{small_cap_count}只(阈值{self.params.market_cap_threshold}亿)")



    def filter_stocks_for_buying(self,candidate_stocks,st_list,current_date):
        """
        使用约束管理器过滤买入候选股票
        """
        if not candidate_stocks:
            return []

        #构建约束检查上下文
        context=self._build_constraint_context(st_list,current_date)

        #使用约束管理器进行买入过滤
        filtered_stocks=self.constraint_manager.filter_for_buying(candidate_stocks,context)

        return filtered_stocks



    def prenext(self):
        """在策略正式开始前的预处理阶段调用"""
        self.next()

    def next(self):
        """
        主要策略逻辑
        每个bar都会调用这个方法
        """
        cur_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        print(f"cur_date:{cur_date}")
        # 增量回测恢复状态
        if self.state_manager.restore_state(self.prev_date, self.to_restore, self.portfolio, self.dividend_tax, self.is_first_run):
            self.is_initialized=True
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
        if cur_date in self.monthly_rebalance_days and self.is_initialized and self.params.enable_rebalance:
            # 在调仓之前，取消订单
            self.trade_executor.cancel_pending_orders()
            # 执行再平衡调仓
            self._rebalance_deal(cur_date, cur_st_list)

        elif cur_date in self.monthly_adjust_days:
            # 在调仓之前，取消订单
            self.trade_executor.cancel_pending_orders()

            #执行Topk策略调仓
            self._topk_adjust(cur_date, cur_st_list)


    def _rebalance_deal(self, current_date, cur_st_list):
        """
        再平衡投资组合，将当前持仓重新平衡到等权重

        参数：
            cur_date: 当前日期
            cur_st_list: 当前ST股票列表
        """
        self.log(f"执行再平衡操作，日期：{current_date}")

        # 获取当前持仓股票
        current_positions = sorted(self.portfolio.get_holding_stocks())
        if not current_positions:
            self.log("当前无持仓，跳过再平衡")
            return

        #过滤掉ST股票
        valid_positions=[stock for stock in current_positions if stock not in cur_st_list]

        stock_dic = get_stock_dic(self.datas)
        current_value = self.broker.getvalue()  # 总资产价值
        # 计算每只股票的目标权重（等权重）
        target_weight = 1.0 / len(valid_positions)

        # 持仓股票权重
        dict_weights_raw = {}

        for stock in valid_positions:
            stock_positions = self.portfolio.get_stock_positions(stock)
            bs_size = sum(pos.size for pos in stock_positions.values())
            bs_price = stock_dic[stock]['price']
            w = (bs_price * bs_size) / (
                        current_value * self.params.max_position)  # 价格*size/(current_value*self.params.max_position)
            dict_weights_raw[stock] = round(w, 8)

        weights_df=pd.DataFrame([
            {'code':stock,'weight':target_weight}
            for stock in valid_positions
        ])
        self.log(f"current_positions:{current_positions}")
        self.log(f"valid_positions:{valid_positions}")
        self.log(f"再平衡股票数量：{len(valid_positions)}")
        self.log(f"历史权重：{dict_weights_raw}")
        self.log(f"目标权重：{weights_df}")

        self.trade_executor.rebalance_portfolio(current_date, weights_df, dict_weights_raw)


    def _topk_adjust(self,cur_date,cur_st_list):
        """
        Topk策略调仓逻辑

        参数：
            cur_date:当前日期
            cur_st_list：当前ST股票列表
        """
        self.log(f"执行Topk策略调仓：{cur_date}")
        #获取当前日期的股票分数数据
        current_scores=self._get_current_scores(cur_date)
        if current_scores.empty:
            self.log(f"警告：{cur_date}没有找到分数数据，跳过调仓")
            return
        #从卖出列表中排除已经被ST逻辑处理的股票
        if not self.is_initialized:
            #初始建仓：选择分数最高的50只股票
            self._initial_position(current_scores,cur_st_list,cur_date)
        else:
            #常规调仓:卖出分数最低的5只，买入分数最高的5只
            self._regular_adjust(current_scores,cur_st_list,cur_date)

    def _get_current_scores(self,cur_date):
        """
        获取当前日期的股票分数数据

        参数：
            cur_date:当前日期

        返回：
            Dataframe:包含code和score列数据
        """
        if self.adjust_data is None:
            return pd.DataFrame()

        #转换日期格式进行匹配
        adjust_data_copy=self.adjust_data.copy()
        adjust_data_copy['date']=pd.to_datetime(adjust_data_copy['date'])
        target_date=pd.to_datetime(cur_date)

        #筛选当前日期的数据
        current_data=adjust_data_copy[adjust_data_copy['date']==target_date]

        if current_data.empty:
            return pd.DataFrame()

        #按分数降序排列
        current_data=current_data.sort_values('score',ascending=False)

        return current_data[['code','score']].reset_index(drop=True)


    def _get_sufficient_candidates(self,current_scores,target_count,cur_st_list,current_date,exclude_stocks=None):
        """动态扩展候选股票范围，确保获得足够数量的有效股票"""
        if exclude_stocks is None:
            exclude_stocks=[]

        #排除指定股票
        available_scores=current_scores[~current_scores['code'].isin(exclude_stocks)].copy()

        if len(available_scores)==0:
            self.log(f"警告：没有可用的候选股票")
            return []

        #从2倍开始，逐步扩大候选范围直到获得足够的有效股票
        multiplier=2
        max_multiplier=5

        while multiplier<=max_multiplier:
            candidate_count = min(target_count*multiplier, len(available_scores))
            candidate_stocks = available_scores.head(candidate_count)['code'].tolist()

            # 使用约束管理器过滤股票
            filtered_stocks = self.filter_stocks_for_buying(candidate_stocks, cur_st_list, current_date)

            self.log(f"尝试{multiplier}倍候选范围，候选{candidate_count}只，过滤后{len(filtered_stocks)}只，目标{target_count}只")

            #如果过滤后的股票数量满足目标，或者已经用完所有可用股票，则停止扩展
            if len(filtered_stocks)>=target_count or candidate_count>=len(available_scores):
                break

            multiplier+=1

        if len(filtered_stocks)<target_count:
            self.log(f"警告：即使扩展到{multiplier}倍候选范围，仍只能获得{len(filtered_stocks)}只有效股票，少于目标{target_count}只")

        # 在返回前确保不超过目标数量
        if len(filtered_stocks) > target_count:
            #按原始分数排序取前N只
            original_order = available_scores['code'].tolist()
            filtered_stocks = [stock for stock in original_order if stock in filtered_stocks][:target_count]

        return filtered_stocks



    def _initial_position(self, current_scores,cur_st_list,current_date):
        """
        初始建仓：选择分数最高的50只股票等权建仓

        参数：
            current_scores:当前分数数据
        """
        self.log(f"执行初始建仓，目标股票数量：{self.params.initial_stocks}")
        #使用动态候选股票选择方法
        filtered_stocks=self._get_sufficient_candidates(
            current_scores,
            self.params.initial_stocks,
            cur_st_list,
            current_date
        )

        #_get_sufficient_candidates已确保不超过目标数量
        top_stocks=filtered_stocks
        #计算等权重
        target_weight=1.0/len(top_stocks)
        #构建权重DataFrame
        weights_df=pd.DataFrame({
            'code':top_stocks,
            'weight':[target_weight]*len(top_stocks)
        })
        self.log(f'初始化建仓股票：{top_stocks}')

        #执行建仓
        self.trade_executor.rebalance_portfolio(current_date, weights_df,{})
        self.is_initialized=True

        #输出初始建仓之后的市值统计
        self.print_market_cap_statistics('初始建仓后持仓',top_stocks)
        self.log(f"初始建仓完成，持仓股票数量：{len(top_stocks)},买入日期：{current_date}")

    def _select_stocks_with_priority(self,holding_scores,actual_sell_count,cur_st_list,current_date):
        """统一的卖出股票选择逻辑 - 不在股票池的股票优先卖出，绕过持有期约束"""
        context = self._build_constraint_context(cur_st_list,current_date)

        #统一处理：先过滤约束，再按优先级排序
        sellable_candidates=[]
        for _,row in holding_scores.iterrows():
            stock=row['code']
            score=row['score']

            # 计算卖出优先级
            priority = self._calculate_sell_priority(stock, context)

            #不在股票池的股票绕过持有期约束，直接可卖
            if priority==0:
                sellable_candidates.append((stock,score,priority))
            else:
                #在股票池的股票检查持有期约束
                can_sell,_ =self.constraint_manager.check_single_constraint(stock,context,'MinHoldingDays')
                if can_sell:
                    sellable_candidates.append((stock,score,priority))

        if not sellable_candidates:
            self.log(f"警告：没有满足约束条件的股票可以卖出")
            return []
        self.log(f"sellable_candidates")
        for i, (stock_code, score, priority) in enumerate(sellable_candidates, 1):
            self.log(f"{i:2d}. {stock_code:10s} | 分数: {score:.6f} | 优先级: {priority}")
        #按优先级排序，同优先级内分数升序（分数低的优先卖出）
        sellable_candidates.sort(key=lambda x: (x[2],x[1]))

        #选择前N只股票
        final_sell_count=min(actual_sell_count,len(sellable_candidates))
        stocks_to_sell=[stock for stock,_,_ in sellable_candidates[:final_sell_count]]

        #统计日志
        self._log_sell_statistics(sellable_candidates,stocks_to_sell)

        return stocks_to_sell


    def _calculate_sell_priority(self, stock, context):
        """
            计算单只股票的卖出优先级

            Returns:
                0:不在股票池(最高优先级)
                1:在股票池可卖出
                0:不可卖出
        """
        # 检查是否在股票池
        in_pool, _ = self.constraint_manager.check_single_constraint(stock, context, 'StockPool')

        if self.params.stock_pool_mode<=0:
            #未启用股票池约束，所有股票同等优先级
            return 1

        #启用股票池约束，不在池内优先卖出
        return 1 if in_pool else 0


    def _log_sell_statistics(self,sellable_candidates,stocks_to_sell):
        """计算卖出统计信息"""
        # 统计不同优先级的股票数量
        if self.params.stock_pool_mode > 0:
            #统计不同优先级数量
            priority_0_count=sum(1 for _,_,p in sellable_candidates if p==0)
            priority_1_count = sum(1 for _, _, p in sellable_candidates if p == 1)
            self.log(f"可卖出股票：不在股票池{priority_0_count}只，在股票池{priority_1_count}")
        else:
            self.log(f"可卖出股票：{len(sellable_candidates)}只")
        self.log(f"实际卖出数量：{len(stocks_to_sell)}")


    def _select_stocks_to_buy(self,current_scores,current_holding,actual_buy_count,cur_st_list,current_date):
        """选择买入股票"""
        # 使用动态候选股票选择方法，排除当前持仓
        filtered_stocks=self._get_sufficient_candidates(
            current_scores,
            actual_buy_count,
            cur_st_list,
            current_date,
            exclude_stocks=current_holding
        )

        # _get_sufficient_candidates已确保不超过调仓数量(考虑ST股票卖出)
        stocks_to_buy = filtered_stocks

        return stocks_to_buy


    def _log_adjust_plan(self,stocks_to_sell,stocks_to_buy,current_holding,st_stock_sell):
        """记录调仓计划日志"""
        self.log(f"计划卖出股票：{stocks_to_sell}")
        self.log(f"计划买入股票：{stocks_to_buy}")

        # 输出调仓前持仓的市值统计
        self.print_market_cap_statistics('调仓前持仓', list(current_holding))

        # 输出卖出和买入股票的市值统计
        if stocks_to_sell:
            self.print_market_cap_statistics('计划卖出', stocks_to_sell)
        if stocks_to_buy:
            self.print_market_cap_statistics('计划买入', stocks_to_buy)

        # 预期调仓后持仓
        expected_holding = (set(current_holding) - set(stocks_to_sell) - set(st_stock_sell)) | set(stocks_to_buy)
        expected_holding_count = len(expected_holding)
        self.log(f"预期调仓后持仓股票数量：{expected_holding_count}")
        self.print_market_cap_statistics('预期调仓后持仓', list(expected_holding))



    def _save_adjust_plan_to_csv(self, stocks_to_sell, stocks_to_buy, current_date, file_path=None):
        """
        将调仓计划保存为CSV文件

        参数:
            stocks_to_sell: 计划卖出的股票列表
            stocks_to_buy: 计划买入的股票列表
            current_date: 当前日期
            file_path: 保存文件路径，默认为None时自动生成
        """
        if file_path is None:
            date_str = bt.num2date(current_date).strftime('%Y%m%d')
            file_path = f"adjust_plan_{date_str}.csv"

        # 创建调仓计划记录列表
        plan_records = []

        # 添加卖出记录
        for stock_code in stocks_to_sell:
            plan_records.append({
                'date': bt.num2date(current_date).strftime('%Y-%m-%d'),
                'operation': 'sell',
                'stock': stock_code,
                'amount': 0,  # 卖出数量为正数
                'price': 0,  # 可选的价格信息
            })

        # 添加买入记录
        for stock_code in stocks_to_buy:
            plan_records.append({
                'date': bt.num2date(current_date).strftime('%Y-%m-%d'),
                'operation': 'buy',
                'stock': stock_code,
                'amount': 0,
                'price': 0,  # 可选的价格信息
            })

        # 转换为DataFrame并保存
        if plan_records:
            df = pd.DataFrame(plan_records)

            # 确保列的顺序一致
            columns = ['date', 'operation', 'stock', 'amount', 'price']
            df = df[columns]

            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            self.log(f"调仓计划已保存到：{file_path}，共{len(df)}条记录")

            # 打印统计信息
            sell_count = len([r for r in plan_records if r['operation'] == 'sell'])
            buy_count = len([r for r in plan_records if r['operation'] == 'buy'])
            self.log(f"卖出 {sell_count} 只股票，买入 {buy_count} 只股票")
        else:
            self.log("无调仓计划需要保存")

        return file_path


    def _regular_adjust(self, current_scores, cur_st_list,current_date):
        """
        常规调仓：卖出持仓中分数最低的5只，买入不在持仓中分数最高的5只

        参数：
            current_scores:当前分数数据
            cur_st_list:ST股票列表
        """
        current_holding=self.portfolio.get_holding_stocks()
        current_holding_count=len(current_holding)
        self.log(f"执行常规调仓，当前持仓数量：{current_holding_count},包含:{current_holding}")

        # st股票卖出列表
        st_stock_sell=[stock for stock in current_holding if stock in cur_st_list]
        # 获取持仓股票分数(排除ST股票)
        non_st_holding=[stock for stock in current_holding if stock not in st_stock_sell]
        holding_scores=current_scores[current_scores['code'].isin(non_st_holding)]
        if len(holding_scores)<self.params.adjust_count:
            self.log(f"警告：持仓股票数量({len(holding_scores)})少于调仓数量({self.params.adjust_count})")

        #计算实际需要调整的数量
        holding_diff=current_holding_count-self.params.initial_stocks
        actual_sell_count = self.params.adjust_count + max(0,holding_diff)
        actual_buy_count = self.params.adjust_count + max(0,-holding_diff)+len(st_stock_sell)
        self.log(f"实际持仓{current_holding_count}只，将卖出{actual_sell_count}只，买入{actual_buy_count}只")

        #根据配置决定是否启用股票池优先级
        #统一的卖出股票选择逻辑，自动处理优先级
        stocks_to_sell=self._select_stocks_with_priority(holding_scores,actual_sell_count,cur_st_list,current_date)
        if not stocks_to_sell:
            constraint_msg=f"最小持有期({self.params.min_holding_days}天)" if self.params.min_holding_days>0 else '其它'
            self.log(f"警告：没有满足{constraint_msg}要求的股票可以卖出")
            return

        #选择买入股票
        stocks_to_buy=self._select_stocks_to_buy(current_scores,current_holding,actual_buy_count,cur_st_list,current_date)

        #打印调仓计划信息
        self._log_adjust_plan(stocks_to_sell, stocks_to_buy, current_holding, st_stock_sell)

        #保存调仓计划到csv文件
        # self._save_adjust_plan_to_csv(stocks_to_sell,stocks_to_buy,current_date)

        #注意： 实际的持仓更新将在notify_order中通过Portfolio对象处理

        stock_dic = get_stock_dic(self.datas)
        current_value = self.broker.getvalue()  # 总资产价值
        # 持仓股票权重
        dict_weights_raw = {}
        for stock in self.portfolio.get_holding_stocks():
            stock_positions=self.portfolio.get_stock_positions(stock)
            bs_size = sum(pos.size for pos in stock_positions.values())
            bs_price = stock_dic[stock]['price']
            w = (bs_price * bs_size) / (current_value * self.params.max_position)  # 价格*size/(current_value*self.params.max_position)
            w = round(w, 8)
            dict_weights_raw[stock] = w

        #执行调仓
        self.trade_executor.adjust_portfolio_topk(current_date,stocks_to_sell,stocks_to_buy,dict_weights_raw,st_stock_sell)



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

                #记录买入交易到调仓记录表
                self.adjust_records.append({
                    'date':str_buy_date,
                    'operation':'buy',
                    'stock':stock,
                    'price':order.executed.price,
                    'amount':order.executed.size
                })

                self.log(
                    f'买单执行,{bt.num2date(order.executed.dt)},股票:{order.data._name},{order.getstatusname()},成交价格:{order.executed.price},成交量:{order.executed.size},手续费:{order.executed.comm}, 创建时间 {bt.num2date(order.created.dt)}')
            elif order.issell():
                # 卖出股票，扣税, 先进先出调整头寸
                stock = order.data._name
                current_date = bt.num2date(order.executed.dt)

                #记录卖出交易到调仓记录表
                str_sell_date=current_date.strftime('%Y-%m-%d')
                self.adjust_records.append({
                    'date': str_sell_date,
                    'operation': 'sell',
                    'stock': stock,
                    'price': order.executed.price,
                    'amount': order.executed.size
                })

                tax = self.dividend_handler.get_dividend_tax(stock, order.executed.size, current_date)

                # 使用Portfolio对象处理卖出（FIFO原则）
                self.portfolio.remove_position(stock, order.executed.size)


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
                f'持仓市值:{self.get_position_value():.2f},现金:{self.broker.getcash():.2f},总收益:{self.total_profit:.2f},胜率:{100 * self.winning_trades / self.total_trades:.2f},盈亏比:{risk_reward_ratio:.2f}')


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

    def get_adjust_records(self):
        """
        获取调仓记录表
        """
        if not self.adjust_records:
            return pd.DataFrame(columns=['date','operation','stock','price','amount'])

        df=pd.DataFrame(self.adjust_records)
        #按日期和操作类型排序，卖出在前，买入在后
        df['operation_order']=df['operation'].map({'sell':0,'buy':1})
        df=df.sort_values(['date','operation_order','stock']).drop('operation_order',axis=1)
        return df.reset_index(drop=True)

    def save_adjust_records(self,file_path=None):
        """
        保存调仓记录表到CSV文件
        """
        if file_path is None:
            file_path=f"adjust_records.csv"

        df=self.get_adjust_records()
        df.to_csv(file_path,index=False,encoding='utf-8-sig')
        self.log(f"调仓记录表已保存到：{file_path},共{len(df)}条记录")
        return file_path


    def stop(self):
        """
        策略结束时执行，计算最终结果并保存状态
        """
        # 使用基准计算器计算策略表现
        self.log(f'len(self.value){len(self.value)}')
        self.benchmark_portfolio_daily = self.benchmark_calculator.calculate_strategy_performance(self.value)
        # 打印回测日收益和基准日收益、收益差值数据
        self.log(f'回测日收益和基准日收益、收益差值数据:{self.benchmark_portfolio_daily}')

        cur_date = self.data0.datetime.date(0).strftime('%Y-%m-%d')

        #保存调仓记录表
        if self.adjust_records:
            self.save_adjust_records(self.filepath)
            self.log(f"调仓记录表统计:共记录{len(self.adjust_records)}笔交易")
        else:
            self.log("无调仓记录")

        # 保存状态
        if self.is_first_run:
            self.state_manager.save_state(
                cur_date, self.benchmark_portfolio_daily, self.portfolio,
                self.dividend_tax, self.reward, self.rewardcnt, self.risk, self.riskcnt
            )

