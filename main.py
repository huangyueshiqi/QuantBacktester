import sys
import time
import argparse

import pandas as pd
import numpy as np
import backtrader as bt
import pickle
import json
import os
from datetime import datetime, date
import glob

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from pred_score import get_pred_scores
from core.strategy import AlphaStrategy
from core.index_strategy import IndexStrategy
from core.topk_strategy import TopkStrategy
from core.buy_strategy import BuyStrategy
from core.llm_strategy import LLMStrategy
from core.analyzer import StrategyAnalyzer
from data.loader import StockDataService, DataPreprocessor
from utils.helpers import (
    rebalancing_day, bt_periods, load_files_by_dates,
    ibd_indicator_cal, plot_strategy_data,
    load_ibt_date, cvxopt, get_st_stock, get_stock_dic_from_dataframe
)
from utils.config import config


class PandasDataExtend(bt.feeds.PandasData):
    """
    扩展的PandasData类

    增加了自定义的数据列用于策略,支持按需配置数据列
    """
    # 增加的自定义线--定义所有可能需要的数据列
    # lines = ('score', 'volume', 'limit', 'stopping', 'tradestatus',
    #          'indecode', 'MARKETVALUE', 'vol5', 'ave5', 'rev5')
    lines = ('score', 'volume', 'limit', 'stopping', 'tradestatus',
             'indecode', 'MARKETVALUE')

    # 各线对应的默认列名，可通过参数覆盖
    params = (
        # 基础参数
        ('datetime', None),  # 使用索引作为日期时间
        ('open', 'open'),  # 开盘价列名
        ('high', 'high'),  # 最高价价列名
        ('low', 'low'),  # 最低价列名
        ('close', 'close'),  # 收盘价列名
        ('openinterest', -1),  # 不使用平仓量

        # 扩展参数
        ('score', 'score'),  # 预测分数列名
        ('volume', 'volume'),  # 成交量列名
        ('limit', -1),  # 涨停价列名
        ('stopping', -1),  # 跌停价列名
        ('tradestatus', -1),  # 交易状态列名
        ('indecode', -1),  # 行业代码列名
        ('MARKETVALUE', -1),  # 市值列名
        # ('vol5', 'vol5'),  # 5日均量列名
        # ('ave5', 'ave5'),  # 5日均价列名
        # ('rev5', 'rev5'),  # 5日成交额均值列名

        # 额外控制参数
        ('use_score', 'True'),  # 是否使用score列
        ('plot', 'False'),  # 是否绘图
    )

    @classmethod
    def from_dataframe(cls, df, name=None, use_score=True, **kwargs):
        """
        从DataFrame创建数据源的便捷工厂方法

        参数：
            df：包含股票数据的DataFrame
            name：数据名称
            use_score: 是否使用score列
            **kwargs： 传递给构造函数PandasDataExtend的其它参数

        返回：
            PandasDataExtend实例
        """
        # 计算技术指标
        # if 'volume' in df.columns and 'close' in df.columns:
        #     df['vol5'] = df['volume'].rolling(window=5, min_periods=1).mean()
        #     df['ave5'] = df['close'].rolling(window=5, min_periods=1).mean()
        #     df['rev5'] = (df['vol5'] * df['ave5']).round(2)

        # 创建数据源实例
        score_param = 'score' if use_score else -1

        data = cls(
            dataname=df,
            datetime=None,  # 日期列是索引列
            score=score_param,
            use_score=use_score,
            plot=kwargs.get('plot', False),
            **{k: v for k, v in kwargs.items() if k != 'plot'}
        )
        return data


class BacktestManager:
    """
    回测管理器

    用于管理和执行回测任务
    """

    def __init__(self, config_file=None):
        """
        初始化回测管理器

        参数:
            config_file: 配置文件路径（可选）
        """

        # 初始化数据服务
        self.data_service = StockDataService()
        self.data_processor = DataPreprocessor()

        # 确保结果目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        paths = [
            config.paths.data,
            config.paths.result,
            config.paths.state,
            config.paths.plot
        ]

        for path in paths:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                    print(f"已创建目录: {path}")
                except Exception as e:
                    print(f"创建目录失败: {path}, 错误: {e}")

    def _prepare_backtest_data(self,start_date,end_date,codes_list,stock_predict_data=None):
        """
        准备回测所需的所有数据

        参数:
            start_date: 开始日期
            end_date: 结束日期
            codes_list: 股票代码列表
            stock_predict_data: 预测分数数据，可选

        返回：
            包含所有处理后数据的字典
        """
        # 获取回测所需的所有数据
        stock_index = config.backtest.benchmark.default
        # 获取指数映射
        index_mapping = config.backtest.benchmark.indices
        index_value = index_mapping.get(stock_index)
        data_dict = self.data_service.get_stock_data_for_backtest(
            start_date, end_date, codes_list, index_value
        )

        # 处理股票数据
        if stock_predict_data is None:
            stock_data = self.data_processor.prepare_stock_data(
                data_dict['stock_detail_data']
            )
        else:
            stock_data = self.data_processor.preprocess_stock_detail_data(
                data_dict['stock_detail_data'], stock_predict_data
            )

        # 处理ST股票数据
        st_dict = self.data_processor.preprocess_st_data(data_dict['st_data'])

        # 处理分红数据
        dividends, dividends_probonus = self.data_processor.check_for_dividends_from_df(
            data_dict['divident_data']
        )

        # 处理拆股数据
        dividends_changert = self.data_processor.load_proright_from_df(
            data_dict['proright_data']
        )

        return {
            'stock_data':stock_data,
            'st_dict':st_dict,
            'dividends':dividends,
            'dividends_probonus':dividends_probonus,
            'dividends_changert':dividends_changert,
            'data_dict':data_dict
        }

    def _setup_cerebro(self,cash=None,add_analyzers=True,plot_output="plot/strategy_plot.png"):
        """
        配置cerebro回测引擎

        参数:
            cash: 初始资金
            commission: 佣金率
            perc: 滑点
            add_analyzers: 是否添加分析器
            plot_output: 绘图输出路径

        返回：
            配置好的cerebro实例
        """
        # 初始化回测引擎
        cerebro = bt.Cerebro()

        # 使用默认配置，如果参数为None
        if cash is None:
            cash = config.backtest.cash
        commission = config.backtest.commission
        perc = config.backtest.slippage

        # 设置初始资金
        cerebro.broker.setcash(cash)
        # 设置佣金
        cerebro.broker.setcommission(commission=commission)
        # 设置滑点
        cerebro.broker.set_slippage_perc(perc=perc)

        # 添加分析器
        if add_analyzers:
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
            cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio', riskfreerate=0.01)  # 夏普比率
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤
            # 添加自定义分析器
            cerebro.addanalyzer(StrategyAnalyzer, _name='my_analyzer', plot_output_path=plot_output)

        return cerebro, cash, commission, perc


    def _load_data_to_cerebro(self,cerebro,data,use_score=True):
        """
        统一的数据加载方法

        参数：
            cerebro:backtrader引擎实例
            data：股票数据Dataframe
            use_score：是否使用score
        """
        cnt_t = 0
        grouped_data = data.groupby(level=0)
        for code, df in grouped_data:
            # 将'datetime'设置为新的索引
            df = df.set_index('datetime')

            # 创建DataFeed对象
            data = PandasDataExtend.from_dataframe(
                df, name=code, use_score=use_score
            )

            cerebro.adddata(data, name=code)
            cnt_t += 1

        return cnt_t


    def get_incremental_date(self,start_date):
        """返回增量回测所需的日期"""
        portfolio_date=start_date     #持仓表状态日期
        backtest_start_date=load_ibt_date(start_date)  #回测开始日期，持仓表日期往前推两个交易日
        return portfolio_date,backtest_start_date


    def run_full_backtest(self, stock_predict_file=None, stock_pool_file=None, plot_output="plot/strategy_plot.png",verbose=True):
        """
        执行全量回测

        参数:
            stock_predict_file: 股票预测数据文件路径（可选）
            stock_pool_file: 股票池文件路径（可选）
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        # 加载股票池
        if stock_pool_file:
            section_codes = pd.read_csv(stock_pool_file)
        else:
            section_codes = pd.read_csv("section_codes.csv")

        section_codes.columns = ['codes']
        sectioncodes = section_codes['codes'].tolist()
        # 加载预测数据
        if stock_predict_file:
            stock_predict_data = pd.read_csv(stock_predict_file, parse_dates=['datetime'])
            stock_predict_data = stock_predict_data.sort_values('datetime')
            # stock_predict_data = stock_predict_data[:200000]
            # start_date = min(stock_predict_data['datetime']).strftime('%Y-%m-%d')
            # end_date = max(stock_predict_data['datetime']).strftime('%Y-%m-%d')
        else:
            stock_predict_data, start_date, end_date = self.data_service.get_stock_predict_data()

        # 保留小数点后八位
        # end_date = '2022-04-28'
        stock_predict_data.columns = ['date', 'code', 'score']
        stock_predict_data['score'] = stock_predict_data['score'].round(8)
        codes_list = stock_predict_data['code'].unique().tolist()

        start_date = config.backtest.strategy.start_date
        end_date = config.backtest.strategy.end_date
        print(f'开始日期：{start_date}，结束日期：{end_date}')

        # 从配置获取调仓参数
        adjust_freq = config.backtest.strategy.adjust_freq
        adjust_day_offset = config.backtest.strategy.adjust_day_offset
        # 加载调仓日期
        rebalancing_days = rebalancing_day(start_date, end_date,adjust_freq,adjust_day_offset)
        print(f"调仓频率：{adjust_freq},调仓日期数量：{len(rebalancing_days)},调仓日期：{rebalancing_days}")

        # 获取回测数据
        backtest_data=self._prepare_backtest_data(start_date,end_date,codes_list, stock_predict_data)
        stock_data=backtest_data['stock_data']
        st_dict=backtest_data['st_dict']
        dividends=backtest_data['dividends']
        dividends_probonus = backtest_data['dividends_probonus']
        dividends_changert = backtest_data['dividends_changert']
        data_dict=backtest_data['data_dict']

        #初始化回测引擎
        cerebro, cash, commission, perc = self._setup_cerebro(plot_output=plot_output)

        # 准备加载到cerebro的数据
        cnt_t=self._load_data_to_cerebro(cerebro,stock_data)

        # 添加策略
        cerebro.addstrategy(
            AlphaStrategy,
            st_dict=st_dict,
            dividends=dividends,
            dividends_probonus=dividends_probonus,
            dividends_changert=dividends_changert,
            BenchmarkDetailData=data_dict['benchmark_detail_data'],
            BaseStockDetailData=data_dict['base_stock_detail_data'],
            startDate=start_date,
            endDate=end_date,
            cash=cash,
            commission=commission,
            perc=perc,
            rebalancing_days=rebalancing_days,
            codeslist=sectioncodes
        )

        print("开始执行回测...")

        # 执行回测
        result = cerebro.run(runonce=False, verbose=verbose)

        # 获取策略实例
        strat = result[0]

        # 输出回测结果
        print("=============== 回测结果 ===============")
        print("年化收益率:", strat.analyzers._AnnualReturn.get_analysis())
        print("夏普比率:", strat.analyzers._SharpeRatio.get_analysis())
        print("最大回撤:", strat.analyzers._DrawDown.get_analysis())
        print("其他指标:", strat.analyzers.my_analyzer.get_analysis())

        return result

    def run_topk_backtest(self, stock_predict_file=None,stock_pool_file=None,topk_adjust=None,plot_output="plot/strategy_plot.png", verbose=True):
        """
        执行全量回测

        参数:
            stock_predict_file: 股票预测数据文件路径（可选）
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        #加载股票池数据
        section_codes=pd.read_csv(stock_pool_file)
        section_codes.columns = ['codes']
        sectioncodes = section_codes['codes'].tolist()

        # 加载预测数据
        if stock_predict_file:
            stock_predict_data = pd.read_csv(stock_predict_file, parse_dates=['datetime'])
            stock_predict_data = stock_predict_data.sort_values('datetime')
            # stock_predict_data = stock_predict_data[:199353]
            # start_date = min(stock_predict_data['datetime']).strftime('%Y-%m-%d')
            # end_date = max(stock_predict_data['datetime']).strftime('%Y-%m-%d')

        saved_state=None
        prev_date=None
        is_first_run=True
        start_date = config.backtest.strategy.start_date
        end_date = config.backtest.strategy.end_date
        if config.backtest.strategy.is_predict:
            is_first_run = False

            prev_date, start_date = self.get_incremental_date(start_date)
            # 将字符串转换为 datetime 日期类型（只保留日期部分）
            start_dt = pd.to_datetime(start_date).normalize()
            end_dt = pd.to_datetime(end_date).normalize()

            # 筛选指定日期范围内的数据
            mask = (stock_predict_data['datetime'].dt.normalize() >= start_dt) & \
                   (stock_predict_data['datetime'].dt.normalize() <= end_dt)
            stock_predict_data = stock_predict_data.loc[mask]

            with open(f'state/state{prev_date}.pkl', 'rb') as f:
                saved_state = pickle.load(f)
                current_value = saved_state['value']

        # 保留小数点后八位
        # end_date = '2025-07-18'
        stock_predict_data.columns = ['date', 'code', 'score']
        stock_predict_data['score'] = stock_predict_data['score'].round(8)
        print(f'开始日期：{start_date}，结束日期：{end_date}')
        codes_list = stock_predict_data['code'].unique().tolist()

        # 处理日期和提取调仓日期
        stock_predict_data['date']=pd.to_datetime(stock_predict_data['date'])
        #从配置获取调仓参数
        adjust_freq=config.backtest.strategy.adjust_freq
        adjust_day_offset=config.backtest.strategy.adjust_day_offset
        #生成调仓日期
        adjust_dates=rebalancing_day(start_date,end_date,adjust_freq,adjust_day_offset)
        #生成再平衡日期（月频）
        rebalancing_days_monthly=rebalancing_day(start_date,end_date,'Q',1)
        print(f"调仓频率：{adjust_freq},调仓日期数量：{len(adjust_dates)},调仓日期：{adjust_dates}")
        print(f"再平衡频率：Q,再平衡日期数量：{len(rebalancing_days_monthly)},再平衡日期：{rebalancing_days_monthly}")

        # 获取回测数据
        backtest_data = self._prepare_backtest_data(start_date, end_date, codes_list, stock_predict_data)
        stock_data = backtest_data['stock_data']
        st_dict = backtest_data['st_dict']
        dividends = backtest_data['dividends']
        dividends_probonus = backtest_data['dividends_probonus']
        dividends_changert = backtest_data['dividends_changert']
        data_dict = backtest_data['data_dict']

        # 初始化回测引擎
        cerebro, cash, commission, perc = self._setup_cerebro(cash=current_value if config.backtest.strategy.is_predict else None,
                                                              add_analyzers=not config.backtest.strategy.is_predict,
                                                              plot_output=plot_output)

        # 准备加载到cerebro的数据
        cnt_t=self._load_data_to_cerebro(cerebro,stock_data)

        # 添加策略
        cerebro.addstrategy(
            TopkStrategy,
            st_dict=st_dict,
            dividends=dividends,
            dividends_probonus=dividends_probonus,
            dividends_changert=dividends_changert,
            BenchmarkDetailData=data_dict['benchmark_detail_data'],
            BaseStockDetailData=data_dict['base_stock_detail_data'],
            start_date=start_date,
            end_date=end_date,
            cash=cash,
            commission=commission,
            perc=perc,
            adjusting_days=adjust_dates,
            codeslist=sectioncodes,
            adjust_data=stock_predict_data,
            rebalancing_days=rebalancing_days_monthly,
            filepath=topk_adjust,
            saved_state=saved_state,
            prev_date=prev_date,
            is_first_run=is_first_run
        )
        print("开始执行回测...")

        # 执行回测
        result = cerebro.run(runonce=False, verbose=verbose)

        # 获取策略实例
        strat = result[0]

        if not config.backtest.strategy.is_predict:
            # 输出回测结果
            print("=============== 回测结果 ===============")
            print("年化收益率:", strat.analyzers._AnnualReturn.get_analysis())
            print("夏普比率:", strat.analyzers._SharpeRatio.get_analysis())
            print("最大回撤:", strat.analyzers._DrawDown.get_analysis())
            print("其他指标:", strat.analyzers.my_analyzer.get_analysis())

        return result

    def run_buy_backtest(self, buy_list_file=None,plot_output="plot/strategy_plot.png", verbose=True):
        """
        执行全量回测

        参数:
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        start_time=time.time()
        #加载买入列表数据
        buy_list = pd.read_csv(buy_list_file)
        # 假设CSV文件有'datetime'和'instrument'列
        # 转换日期列为datetime类型
        start_date = config.backtest.strategy.start_date
        end_date = config.backtest.strategy.end_date
        buy_list['datetime'] = pd.to_datetime(buy_list['datetime'])
        buy_list = buy_list[buy_list['datetime'] < end_date]
        # current_date='2014-04-24'
        # today_buy_signals = buy_list[buy_list['datetime'] == pd.to_datetime(current_date)]
        print(f"成功读取购买列表，共{len(buy_list)}条记录")
        print(buy_list.head())

        # 保留小数点后八位
        # end_date = '2025-07-18'
        sectioncodes=buy_list['instrument'].unique().tolist()
        print(f'开始日期：{start_date}，结束日期：{end_date}')

        # 获取回测数据
        backtest_data = self._prepare_backtest_data(start_date, end_date, sectioncodes)
        stock_data = backtest_data['stock_data']
        st_dict = backtest_data['st_dict']
        dividends = backtest_data['dividends']
        dividends_probonus = backtest_data['dividends_probonus']
        dividends_changert = backtest_data['dividends_changert']
        data_dict = backtest_data['data_dict']

        # 初始化回测引擎
        cerebro, cash, commission, perc = self._setup_cerebro(cash=100000000,
                                                              plot_output=plot_output)

        # 准备加载到cerebro的数据
        # stock_data=stock_data.fillna(0)
        cnt_t=self._load_data_to_cerebro(cerebro,stock_data,use_score=False)
        print(f"加载数据完成：花费{time.time()-start_time:.4f}s")

        # 添加策略
        cerebro.addstrategy(
            BuyStrategy,
            st_dict=st_dict,
            dividends=dividends,
            dividends_probonus=dividends_probonus,
            dividends_changert=dividends_changert,
            BenchmarkDetailData=data_dict['benchmark_detail_data'],
            BaseStockDetailData=data_dict['base_stock_detail_data'],
            start_date=start_date,
            end_date=end_date,
            cash=cash,
            commission=commission,
            perc=perc,
            buy_list=buy_list,
            codeslist=sectioncodes,
        )
        print("开始执行回测...")

        # 执行回测
        result = cerebro.run(verbose=verbose)
        # result = cerebro.run(runonce=False, preload=False, exactbars=False)

        # 获取策略实例
        strat = result[0]
        print('Final Portfolio Value: %.3f' % cerebro.broker.getvalue())

        if not config.backtest.strategy.is_predict:
            # 输出回测结果
            print("=============== 回测结果 ===============")
            print("年化收益率:", strat.analyzers._AnnualReturn.get_analysis())
            print("夏普比率:", strat.analyzers._SharpeRatio.get_analysis())
            print("最大回撤:", strat.analyzers._DrawDown.get_analysis())
            print("其他指标:", strat.analyzers.my_analyzer.get_analysis())

        return result


    def run_llm_backtest(self, trade_file=None,plot_output="plot/strategy_plot.png", verbose=True, metrics_output=None):
        """
        执行全量回测

        参数:
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        start_time=time.time()
        #加载买入列表数据
        trade_list = pd.read_csv(trade_file)
        # 假设CSV文件有'datetime'和'instrument'列
        # 转换日期列为datetime类型
        start_date = config.backtest.strategy.start_date
        end_date = config.backtest.strategy.end_date
        if 'datetime' not in trade_list.columns:
            trade_list.rename(columns={'date': 'datetime', 'stock_code': "instrument"}, inplace=True)
            trade_list=trade_list[['datetime','instrument','score','rank']]
            trade_list = (
                trade_list
                .sort_values(['datetime', 'score'], ascending=[True, False])
                .groupby('datetime')
                .head(10)
            )
        trade_list['datetime'] = pd.to_datetime(trade_list['datetime'])
        # start_date=min(trade_list['datetime']).strftime('%Y-%m-%d')
        trade_list = trade_list[trade_list['datetime'] < end_date]
        # current_date='2014-04-24'
        # today_buy_signals = buy_list[buy_list['datetime'] == pd.to_datetime(current_date)]
        print(f"成功读取交易列表，共{len(trade_list)}条记录")
        print(trade_list.head())

        # 保留小数点后八位
        # end_date = '2025-07-18'
        sectioncodes=trade_list['instrument'].unique().tolist()
        print(f'开始日期：{start_date}，结束日期：{end_date}')

        # 获取回测数据
        backtest_data = self._prepare_backtest_data(start_date, end_date, sectioncodes)
        stock_data = backtest_data['stock_data']
        st_dict = backtest_data['st_dict']
        dividends = backtest_data['dividends']
        dividends_probonus = backtest_data['dividends_probonus']
        dividends_changert = backtest_data['dividends_changert']
        data_dict = backtest_data['data_dict']

        # 初始化回测引擎
        cerebro, cash, commission, perc = self._setup_cerebro(cash=100000000,
                                                              plot_output=plot_output)

        # 准备加载到cerebro的数据
        # stock_data=stock_data.fillna(0)
        cnt_t=self._load_data_to_cerebro(cerebro,stock_data,use_score=False)
        print(f"加载数据完成：花费{time.time()-start_time:.4f}s")

        # 添加策略
        cerebro.addstrategy(
            LLMStrategy,
            st_dict=st_dict,
            dividends=dividends,
            dividends_probonus=dividends_probonus,
            dividends_changert=dividends_changert,
            BenchmarkDetailData=data_dict['benchmark_detail_data'],
            BaseStockDetailData=data_dict['base_stock_detail_data'],
            start_date=start_date,
            end_date=end_date,
            cash=cash,
            commission=commission,
            perc=perc,
            trade_list=trade_list,
            codeslist=sectioncodes,
        )
        print("开始执行回测...")

        # 执行回测
        cerebro.addobserver(bt.observers.BuySell)
        result = cerebro.run(verbose=verbose)
        # result = cerebro.run(runonce=False, preload=False, exactbars=False)

        # 获取策略实例
        strat = result[0]
        final_value = float(cerebro.broker.getvalue())
        print('Final Portfolio Value: %.3f' % final_value)

        if not config.backtest.strategy.is_predict:
            annual_return = strat.analyzers._AnnualReturn.get_analysis()
            sharpe_ratio = strat.analyzers._SharpeRatio.get_analysis()
            drawdown = strat.analyzers._DrawDown.get_analysis()
            other_metrics = strat.analyzers.my_analyzer.get_analysis()

            print("=============== 回测结果 ===============")
            print("年化收益率:", annual_return)
            print("夏普比率:", sharpe_ratio)
            print("最大回撤:", drawdown)
            print("其他指标:", other_metrics)

            if metrics_output is None:
                trade_tag = os.path.splitext(os.path.basename(trade_file or "llm_trade"))[0]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                metrics_output = os.path.join(config.paths.result, f"llm_backtest_metrics_{trade_tag}_{ts}.json")

            def _json_default(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                return str(obj)

            metrics_payload = {
                "trade_file": trade_file,
                "start_date": start_date,
                "end_date": end_date,
                "final_portfolio_value": final_value,
                "annual_return": annual_return,
                "sharpe_ratio": sharpe_ratio,
                "drawdown": drawdown,
                "other_metrics": other_metrics,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }

            with open(metrics_output, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2, default=_json_default)

            strat.metrics_output_path = metrics_output
            strat.metrics_payload = metrics_payload

        # fig=cerebro.plot(style='candle',volume=False,subplot=True)
        # fig[0][0].savefig('plot/backtest_result.png',dpi=300,bbox_inches='tight')

        return result



    def run_incremental_backtest(self, stock_predict_file=None, stock_pool_file=None, plot_output="plot/strategy_plot.png",verbose=True):
        """
        执行增量回测

        参数:
            stock_predict_file: 股票预测数据文件路径（可选）
            stock_pool_file: 股票池文件路径（可选）
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        # 加载股票池
        if stock_pool_file:
            section_codes = pd.read_csv(stock_pool_file)
        else:
            section_codes = pd.read_csv("section_codes.csv")

        section_codes.columns = ['codes']
        sectioncodes = section_codes['codes'].tolist()

        # 加载预测数据
        if stock_predict_file:
            stock_predict_data = pd.read_csv(stock_predict_file, parse_dates=['datetime'])
            # start_date = min(stock_predict_data['datetime']).strftime('%Y-%m-%d')
            # end_date = max(stock_predict_data['datetime']).strftime('%Y-%m-%d')
        else:
            stock_predict_data, start_date, end_date = self.data_service.get_stock_predict_data()

        # 保留小数点后八位
        # end_date = '2022-04-28'
        stock_predict_data.columns = ['date', 'code', 'score']
        stock_predict_data['score'] = stock_predict_data['score'].round(8)
        codes_list = stock_predict_data['code'].unique().tolist()

        start_date = config.backtest.strategy.start_date
        end_date = config.backtest.strategy.end_date
        print(f'开始日期：{start_date}，结束日期：{end_date}')

        # 加载调仓日期
        rebalancing_days = rebalancing_day(start_date, end_date)

        # 获取回测数据
        backtest_data = self._prepare_backtest_data(start_date, end_date, codes_list, stock_predict_data)
        stock_data = backtest_data['stock_data']
        st_dict = backtest_data['st_dict']
        dividends = backtest_data['dividends']
        dividends_probonus = backtest_data['dividends_probonus']
        dividends_changert = backtest_data['dividends_changert']
        data_dict = backtest_data['data_dict']

        # 获取回测周期
        periods = bt_periods(rebalancing_days, start_date, end_date)
        print(f'划分的回测周期:{periods}')

        # 执行增量回测
        cash = config.backtest.cash
        commission = config.backtest.commission
        perc = config.backtest.slippage

        # 初始化变量
        saved_state = None
        positions = {}
        current_value = cash
        is_first_run = True
        end_date_list = []

        for period in periods:
            start, end = period
            prev_date = start
            end_date_list.append(end)
            print(f'回测周期:{start}-{end}')

            cerebro = bt.Cerebro()

            # 初始化broker
            if not is_first_run:
                start = load_ibt_date(start)
                with open(f'state/state{prev_date}.pkl', 'rb') as f:
                    saved_state = pickle.load(f)
                    current_value = saved_state['value']
                with open(f'state/positions{prev_date}.json', 'r') as f:
                    positions = json.load(f)
            # 根据日期进行筛选处理股票行情明细数据
            period_stock_data = stock_data[(stock_data['datetime'] >= start) & (stock_data['datetime'] <= end)]
            # 根据日期进行筛选处理基准数据
            BenchmarkDetailData = data_dict['benchmark_detail_data']
            benchmarkData = BenchmarkDetailData[
                (BenchmarkDetailData['TRADEDATE'] >= start) & (BenchmarkDetailData['TRADEDATE'] <= end)]
            BaseStockDetailData = data_dict['base_stock_detail_data']
            baseStockData = BaseStockDetailData[
                (BaseStockDetailData['TRADEDATE'] >= start) & (BaseStockDetailData['TRADEDATE'] <= end)]

            stock_dict = get_stock_dic_from_dataframe(period_stock_data, prev_date)
            cur_st_list = get_st_stock(st_dict, prev_date)
            # 使用优化器计算权重
            max_position = config.backtest.strategy.max_position
            weights, dict_weights_raw = cvxopt(prev_date, cur_st_list, codes_list, baseStockData,
                                               current_value, stock_dict, positions, max_position)
            if is_first_run:
                prev_date = None

            # 合并股票列表，确保包含所有需要的股票
            stock_list = list(weights['code']) + list(positions.keys())
            stock_list = list(set(stock_list))

            # 只加载这些股票的数据
            period_stock_data = period_stock_data[period_stock_data.index.isin(stock_list)]

            # 加载数据到cerebro
            cnt_t=self._load_data_to_cerebro(cerebro,period_stock_data)

            # 设置佣金和滑点
            cerebro.broker.setcash(current_value)
            cerebro.broker.setcommission(commission=commission)
            cerebro.broker.set_slippage_perc(perc=perc)

            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
            cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
            # cerebro.addanalyzer(StrategyAnalyzer, _name='my_analyzer',plot_output_path="plot/strategy_plot.png")

            # 添加策略
            cerebro.addstrategy(
                AlphaStrategy,
                st_dict=st_dict,
                dividends=dividends,
                dividends_probonus=dividends_probonus,
                dividends_changert=dividends_changert,
                BenchmarkDetailData=benchmarkData,
                BaseStockDetailData=baseStockData,
                startDate=start,
                endDate=end,
                prev_date=prev_date,
                cash=cash,
                commission=commission,
                perc=perc,
                saved_state=saved_state,
                rebalancing_days=rebalancing_days,
                is_first_run=is_first_run,
                codeslist=codes_list,
                weights=weights,
                dict_weights_raw=dict_weights_raw
            )

            # 执行回测
            result = cerebro.run(runonce=False, verbose=verbose)

            # 更新状态
            is_first_run = False

        # 计算最终结果
        combined_df = load_files_by_dates(end_date_list)
        plot_df, metrics = ibd_indicator_cal(
            start_date, end_date, combined_df,
            data_dict['benchmark_detail_data'], end_date_list, cash
        )

        print(f'指标字典:{metrics}')
        # 绘制回测结果图
        plot_strategy_data(start_date, plot_df)

        return metrics


    def run_rebalanceTable_backtest(self, rebalance_data=None,plot_output="plot/strategy_plot.png", verbose=True):
        """
        基于调仓表文件进行等权回测

        参数:
            rebalance_data: 调仓表数据
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        rebalance_data.columns = ['date', 'stock', 'weight']
        # rebalance_data=rebalance_data[:500]

        # 提取调仓日期和股票代码
        rebalance_data.loc[:, 'date'] = pd.to_datetime(rebalance_data['date'])
        rebalance_days = rebalance_data['date'].dt.strftime('%Y-%m-%d').unique().tolist()
        start_date = min(rebalance_data['date']).strftime('%Y-%m-%d')
        end_date = max(rebalance_data['date']).strftime('%Y-%m-%d')
        # end_date='2025-04-13'

        print(f'开始日期：{start_date}，结束日期：{end_date}')
        print(f'调仓日期列表：{rebalance_days}')

        # 提取所有股票代码，用于加载数据
        all_stocks = rebalance_data['stock'].unique().tolist()
        print(f'股票池大小：{len(all_stocks)}')

        # 获取回测数据
        backtest_data = self._prepare_backtest_data(start_date, end_date, all_stocks)
        stock_data = backtest_data['stock_data']
        st_dict = backtest_data['st_dict']
        dividends = backtest_data['dividends']
        dividends_probonus = backtest_data['dividends_probonus']
        dividends_changert = backtest_data['dividends_changert']
        data_dict = backtest_data['data_dict']

        # 初始化回测引擎
        cerebro, cash, commission, perc = self._setup_cerebro(plot_output=plot_output)

        # 准备加载到cerebro的数据
        cnt_t=self._load_data_to_cerebro(cerebro,stock_data,use_score=False)

        print("加载股票数据完成")

        # 创建自定义调仓计划
        # 为每个调仓日创建对应的股票列表
        rebalance_plan = {}
        for date in rebalance_days:
            day_data = rebalance_data[rebalance_data['date'].dt.strftime('%Y-%m-%d') == date]
            # 构造{股票：权重}的子字典
            stock_weights = {}
            for _, row in day_data.iterrows():
                stock_weights[row['stock']] = row['weight']
            rebalance_plan[date] = stock_weights

        # 添加策略
        cerebro.addstrategy(
            AlphaStrategy,
            st_dict=st_dict,
            dividends=dividends,
            dividends_probonus=dividends_probonus,
            dividends_changert=dividends_changert,
            BenchmarkDetailData=data_dict['benchmark_detail_data'],
            BaseStockDetailData=data_dict['base_stock_detail_data'],
            startDate=start_date,
            endDate=end_date,
            cash=cash,
            commission=commission,
            perc=perc,
            rebalancing_days=rebalance_days,
            codeslist=all_stocks,
            rebalance_plan=rebalance_plan,  # 传入调仓计划
            equal_weight=True  # 使用等权重分配
        )

        print("开始执行回测...")

        # 执行回测
        cerebro.broker.set_coc(True)
        result = cerebro.run(runonce=False, verbose=verbose)

        # 获取策略实例
        strat = result[0]

        # 输出回测结果
        print("=============== 回测结果 ===============")
        print("年化收益率:", strat.analyzers._AnnualReturn.get_analysis())
        print("夏普比率:", strat.analyzers._SharpeRatio.get_analysis())
        print("最大回撤:", strat.analyzers._DrawDown.get_analysis())
        print("其他指标:", strat.analyzers.my_analyzer.get_analysis())

        return result

    def run_indexRebalanceTable_backtest(self, rebalance_data=None, plot_output="plot/strategy_plot.png",verbose=True):
        """
        基于指数调仓表文件进行等权回测

        参数:
            rebalance_data: 调仓表数据
            verbose: 是否输出详细信息

        返回:
            回测结果
        """
        rebalance_data.columns = ['date', 'indexcode', 'weight']
        # rebalance_data=rebalance_data[:100]

        # 提取调仓日期和股票代码
        rebalance_data['date'] = pd.to_datetime(rebalance_data['date'])
        rebalance_days = rebalance_data['date'].dt.strftime('%Y-%m-%d').unique().tolist()
        start_date = min(rebalance_data['date']).strftime('%Y-%m-%d')
        end_date = max(rebalance_data['date']).strftime('%Y-%m-%d')
        end_date = '2025-04-16'

        print(f'开始日期：{start_date}，结束日期：{end_date}')
        print(f'调仓日期列表：{rebalance_days}')

        # 提取所有指数代码，用于加载数据
        all_indexcodes = rebalance_data['indexcode'].unique().tolist()
        print(f'指数池大小：{len(all_indexcodes)}')

        # 获取回测所需的所有数据
        stock_index = config.backtest.benchmark.default
        # 获取指数映射
        index_mapping = config.backtest.benchmark.indices
        index_value = index_mapping.get(stock_index)
        data_dict = self.data_service.get_indexcode_data_for_backtest(
            start_date, end_date, index_value
        )

        # 处理股票数据
        index_data = self.data_processor.prepare_indexcode_data(
            data_dict['indexcode_data']
        )
        # 初始化回测引擎
        cerebro, cash, commission, perc = self._setup_cerebro(plot_output=plot_output)

        # 准备加载到cerebro的数据
        cnt_t=self._load_data_to_cerebro(cerebro,index_data,use_score=False)

        # 创建自定义调仓计划
        # 为每个调仓日创建对应的股票列表
        rebalance_plan = {}
        for date in rebalance_days:
            day_data = rebalance_data[rebalance_data['date'].dt.strftime('%Y-%m-%d') == date]
            # 构造{股票：权重}的子字典
            stock_weights = {}
            for _, row in day_data.iterrows():
                stock_weights[row['indexcode']] = row['weight']
            rebalance_plan[date] = stock_weights

        # 添加策略
        cerebro.addstrategy(
            IndexStrategy,
            BenchmarkDetailData=data_dict['benchmark_detail_data'],
            startDate=start_date,
            endDate=end_date,
            cash=cash,
            commission=commission,
            perc=perc,
            rebalancing_days=rebalance_days,
            rebalance_plan=rebalance_plan,  # 传入调仓计划
            equal_weight=True  # 使用等权重分配
        )

        print("开始执行回测...")

        # 执行回测
        result = cerebro.run(runonce=False, verbose=verbose)

        # 获取策略实例
        strat = result[0]

        # 输出回测结果
        print("=============== 回测结果 ===============")
        print("年化收益率:", strat.analyzers._AnnualReturn.get_analysis())
        print("夏普比率:", strat.analyzers._SharpeRatio.get_analysis())
        print("最大回撤:", strat.analyzers._DrawDown.get_analysis())
        print("其他指标:", strat.analyzers.my_analyzer.get_analysis())

        return result


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='量化交易回测系统')

    # 添加参数
    parser.add_argument('--mode', type=str, default='topk',
                        choices=['full', 'topk', 'incremental', 'rebalanceTable', 'index_rebalanceTable','buy','llm'],
                        help='回测模式：full(全量回测),incremental(增量回测),rebalanceTable(基于调仓表文件回测),index_rebalanceTable(基于指数调仓表文件回测)')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--predict', type=str, default='input/prediction_results.csv', help='预测数据文件路径')
    parser.add_argument('--pool', type=str, default='input/section_codes1.csv', help='股票池文件路径')
    parser.add_argument('--rebalance_file', type=str, default='./input/25071501trade_log.csv',
                        help='股票调仓表文件路径')  # premium_value_strategy_equal_weight.csv,250625trade_log
    parser.add_argument('--index_rebalance_file', type=str, default='./input/250417stock_gai.csv',
                        help='指数调仓表文件路径')
    parser.add_argument('--output', type=str, default='', help='输出结果文件路径')  # output.txt
    parser.add_argument('--topk_adjust', type=str, default='adjust_records.csv', help='topk调仓结果文件路径')
    parser.add_argument('--plot_output',type=str,default='plot/strategy_plot1.png',help='策略回测图片保存路径')
    parser.add_argument('--trade_file', type=str, default='', help='llm模式交易信号CSV文件路径（可选）')
    parser.add_argument('--trade_dir', type=str, default='', help='llm模式交易信号CSV目录（可选，目录下将按*.csv批量执行）')
    parser.add_argument('--metrics_output', type=str, default='', help='llm模式回测指标JSON输出路径（单文件模式使用，可选）')
    parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')

    # 解析参数
    args = parser.parse_args()

    # 初始化回测管理器
    backtest_manager = BacktestManager(args.config)

    # 设置输出文件
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='utf-8')

    # 记录开始时间
    start_time = time.time()

    # 根据模式执行回测
    if args.mode == 'full':
        print("执行全量回测...")
        backtest_manager.run_full_backtest(args.predict, args.pool,args.plot_output, args.verbose)
    elif args.mode == 'topk':
        print("执行TOPK全量回测...")
        # pred_score = get_pred_scores("gru_company", True)
        backtest_manager.run_topk_backtest(args.predict, args.pool, args.topk_adjust,args.plot_output,args.verbose)
    elif args.mode == 'buy':
        print("执行buy全量回测...")
        buy_list_file='/home/quant/zc/backtrader/QuantBacktester_57/new_buy_df_30.csv'
        backtest_manager.run_buy_backtest(buy_list_file,args.plot_output,args.verbose)
    elif args.mode == 'llm':
        print("执行llm-stock全量回测...")
        metrics_output = args.metrics_output if args.metrics_output else None
        if args.trade_dir:
            trade_files = sorted(glob.glob(os.path.join(args.trade_dir, "*.csv")))
            if not trade_files:
                raise FileNotFoundError(f'目录下未找到csv文件：{args.trade_dir}')
            for tf in trade_files:
                backtest_manager.run_llm_backtest(tf, args.plot_output, args.verbose, metrics_output=None)
        else:
            trade_file = args.trade_file or '/home/quant/zc/backtrader/QuantBacktester_57/llm/trade_portfolio_24_0.csv'
            backtest_manager.run_llm_backtest(trade_file, args.plot_output, args.verbose, metrics_output=metrics_output)
    elif args.mode == 'incremental':  # incremental
        print("执行增量回测...")
        backtest_manager.run_incremental_backtest(args.predict, args.pool, args.plot_output,args.verbose)
    elif args.mode == 'rebalanceTable':
        print("执行基于调仓表文件回测...")
        # 加载调仓表数据
        if not os.path.exists(args.rebalance_file):
            raise FileNotFoundError(f'调仓表文件不存在：{args.rebalance_file}')

        # rebalance_data=pd.read_csv(args.rebalance_file,dtype={'SYMBOL':str})
        rebalance_data = pd.read_csv(args.rebalance_file, parse_dates=['date'])
        # rebalance_data=rebalance_data[:2000]
        backtest_manager.run_rebalanceTable_backtest(rebalance_data, args.plot_output,args.verbose)
    elif args.mode == 'index_rebalanceTable':
        print("执行基于指数调仓表文件回测...")
        # 加载调仓表数据
        if not os.path.exists(args.index_rebalance_file):
            raise FileNotFoundError(f'调仓表文件不存在：{args.index_rebalance_file}')

        rebalance_data = pd.read_csv(args.index_rebalance_file, parse_dates=['TRADE_DT'])
        backtest_manager.run_indexRebalanceTable_backtest(rebalance_data,args.plot_output, args.verbose)

    # 输出总运行时间
    end_time = time.time()
    print(f'回测总运行时间：{end_time - start_time:.2f}秒')

    # 关闭输出文件
    if args.output:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
