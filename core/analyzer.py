import backtrader as bt
import pandas as pd
import numpy as np
import empyrical as er
from abc import ABC, abstractmethod

# 修复 NINF 问题
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf

from utils.helpers import plot_strategy_data,plot_strategy_with_multiple_funds

class BaseAnalyzer(ABC):
    """
    分析器抽象基类
    
    定义了策略分析器的通用接口，遵循单一职责原则和开闭原则
    子类可以实现不同的分析方法而不影响外部调用
    """
    
    @abstractmethod
    def analyze(self, strategy):
        """
        分析策略性能
        
        参数:
            strategy: 要分析的策略对象
            
        返回:
            分析结果字典
        """
        pass


class BaseAnalyzerImpl(BaseAnalyzer):
    """
    分析器抽象基类

    定义了策略分析器的通用接口，遵循单一职责原则和开闭原则
    子类可以实现不同的分析方法而不影响外部调用
    """

    def analyze(self, strategy):
        """
        分析策略性能

        参数:
            strategy: 要分析的策略对象

        返回:
            分析结果字典
        """
        pass


class StrategyAnalyzer(bt.Analyzer):
    """
    策略分析器
    
    用于分析交易策略的性能，计算各种指标
    继承自bt.Analyzer以便集成到backtrader框架
    """
    
    def __init__(self,plot_output_path="plot/strategy_plot.png"):
        """初始化分析器"""
        super(StrategyAnalyzer, self).__init__()
        self.base_analyzer = BaseAnalyzerImpl()
        self.plot_out_path=plot_output_path
        # 交易统计数据
        self.wins = 0  # 盈利次数
        self.losses = 0  # 亏损次数
        self.gross_profits = 0  # 盈利金额
        self.gross_losses = 0  # 亏损金额
        self.riskfreerate=0.01
    
    def next(self):
        """每个时间点执行的逻辑"""
        pass
    
    def monthly_analysis(self):
        """
        计算每月的收益差
        
        返回:
            每月收益差列表
        """
        # 提取日期和月份
        data = self.strategy.benchmark_portfolio_daily
        dates = pd.to_datetime(data.index)
        years=dates.year
        months = dates.month  # 转换为月份

        # 初始化存储结果
        monthly_return_diffs = []

        # 获取每日收益率和基准收益率
        daily_returns = data['value_daily_ratio'].values
        daily_benchmark_returns = data['daily_return'].values

        # 按年份和月份分组计算
        # 创建(year,month)对的列表并去重排序
        year_month_pairs=sorted(list(set(zip(years,months))))

        for year,month in year_month_pairs:
            # 获取当前年月的索引
            mask = (years==year)&(months == month)
            valid_daily_returns = daily_returns[mask]
            valid_benchmark_returns = daily_benchmark_returns[mask]
            # 过滤掉NaN值
            valid_daily_returns = valid_daily_returns[~np.isnan(valid_daily_returns)]
            valid_benchmark_returns = valid_benchmark_returns[~np.isnan(valid_benchmark_returns)]
            #如果当月没有有效数据，跳过
            if len(valid_daily_returns)==0:
                continue
            # 计算收益率和差异
            monthly_return = np.prod(1 + valid_daily_returns) - 1
            monthly_benchmark_return = np.prod(1 + valid_benchmark_returns) - 1
            monthly_return_diff = monthly_return - monthly_benchmark_return
            monthly_return_diffs.append(monthly_return_diff)

        print(f'每月的收益差monthly_return_diffs{monthly_return_diffs}')
        return monthly_return_diffs

    def yearly_analysis(self):
        """
        计算每年的收益差
        
        返回:
            每年收益差列表
        """
        # 提取日期和年份
        data = self.strategy.benchmark_portfolio_daily
        dates =  pd.to_datetime(data.index)
        years = dates.year  # 转换为年份

        # 初始化存储结果
        yearly_return_diffs = []

        # 获取每日收益率和基准收益率
        daily_returns = data['value_daily_ratio'].values
        daily_benchmark_returns = data['daily_return'].values
        
        # 按年份分组计算
        unique_years = sorted(list(set(years)))
        for year in unique_years:
            # 获取当前年份的索引
            year_mask = (years == year)
            valid_daily_returns = daily_returns[year_mask]
            valid_benchmark_returns = daily_benchmark_returns[year_mask]
            # 过滤掉NaN值
            valid_daily_returns = valid_daily_returns[~np.isnan(valid_daily_returns)]
            valid_benchmark_returns = valid_benchmark_returns[~np.isnan(valid_benchmark_returns)]
            if len(valid_daily_returns)==0:
                continue
            # 计算收益率和差异
            yearly_return = np.prod(1 + valid_daily_returns) - 1
            yearly_benchmark_return = np.prod(1 + valid_benchmark_returns) - 1
            yearly_return_diff = yearly_return - yearly_benchmark_return
            yearly_return_diffs.append(yearly_return_diff)

        print(f'每年的收益差yearly_return_diffs{yearly_return_diffs}')
        return yearly_return_diffs
    
    def stop(self):
        """
        策略结束时计算并保存各种性能指标
        """
        # 总的交易次数（买卖配对次数）
        total_trades = self.wins + self.losses
        # 交易胜率 = 盈利交易次数 / 总的交易次数
        win_ratio = self.wins / total_trades if total_trades > 0 else 0
        # 平均盈利
        avg_profit = self.gross_profits / self.wins if self.wins > 0 else 0
        # 平均亏损
        avg_loss = self.gross_losses / self.losses if self.losses > 0 else 0
        # 盈亏比 = 盈利平均值 / 亏损平均值
        if avg_loss != 0:
            risk_reward_ratio = avg_profit / abs(avg_loss)
        else:
            risk_reward_ratio = 0

        # 策略回测数据
        benchmark_portfolio_daily = self.strategy.benchmark_portfolio_daily

        # 计算平均超额收益
        mean_excess_return = np.mean(benchmark_portfolio_daily['diff'])
        # 计算超额收益的标准差
        std_excess_return = np.std(benchmark_portfolio_daily['diff'])
        # 计算信息比率 = 平均超额收益 / 超额收益的标准差
        information_ratio = mean_excess_return / std_excess_return if std_excess_return > 0 else 0

        #计算日化无风险利率
        risk_free_daily=(1+self.riskfreerate)**(1/252)-1

        # 计算各种指标
        alpha = er.stats.alpha(benchmark_portfolio_daily['value_daily_ratio'],
                               benchmark_portfolio_daily['daily_return'],
                               risk_free=risk_free_daily)
        beta = er.stats.beta(benchmark_portfolio_daily['value_daily_ratio'],
                             benchmark_portfolio_daily['daily_return'],
                             risk_free=risk_free_daily)
        sharpe_ratio = er.stats.sharpe_ratio(benchmark_portfolio_daily['value_daily_ratio'], risk_free=risk_free_daily)
        omega_ratio = er.stats.omega_ratio(benchmark_portfolio_daily['value_daily_ratio'], risk_free=risk_free_daily)
        sortino_ratio = er.stats.sortino_ratio(benchmark_portfolio_daily['value_daily_ratio'])
        base_annual_return = er.stats.annual_return(benchmark_portfolio_daily['daily_return'])
        annual_return = er.stats.annual_return(benchmark_portfolio_daily['value_daily_ratio'])
        cum_returns_final = er.stats.cum_returns_final(benchmark_portfolio_daily['value_daily_ratio'])
        volatility = er.stats.annual_volatility(benchmark_portfolio_daily['value_daily_ratio'])
        calmar_ratio = er.stats.calmar_ratio(benchmark_portfolio_daily['value_daily_ratio'])
        max_drawdown = er.stats.max_drawdown(benchmark_portfolio_daily['value_daily_ratio'])
        max_diff_drawdown = er.stats.max_drawdown(benchmark_portfolio_daily['diff'])

        # 与benchmark比的胜率
        benchmark_portfolio_daily['win_loss_lst'] = [int(e > 0) for e in benchmark_portfolio_daily['diff']]
        daily_wins = sum(1 for item in benchmark_portfolio_daily['win_loss_lst'] if item == 1)
        daily_win_ratio_benchmark = daily_wins / len(benchmark_portfolio_daily['win_loss_lst'])

        # 月度和年度胜率计算
        monthly_return_diffs = self.monthly_analysis()
        yearly_return_diffs = self.yearly_analysis()
        monthly_wins = sum(1 for item in monthly_return_diffs if item > 0)
        yearly_wins = sum(1 for item in yearly_return_diffs if item > 0)
        monthly_win_ratio_benchmark = monthly_wins / len(monthly_return_diffs) if len(monthly_return_diffs) > 0 else 0
        yearly_win_ratio_benchmark = yearly_wins / len(yearly_return_diffs) if len(yearly_return_diffs) > 0 else 0

        # 输出指标数据赋值
        self.rets['datas'] = {
            'max_drawdown': round(max_drawdown, 3),  # 组合的最大回撤
            'max_diff_drawdown': round(max_diff_drawdown, 3),  # 超额收益的最大回撤
            'daily_win_ratio_benchmark': round(daily_win_ratio_benchmark, 3),  # 与benchmark比较的日胜率
            'monthly_win_ratio_benchmark': round(monthly_win_ratio_benchmark, 3),  # 与benchmark比较的月胜率
            'yearly_win_ratio_benchmark': round(yearly_win_ratio_benchmark, 3),  # 与benchmark比较的年胜率
            'win_ratio': round(win_ratio, 3),  # 交易胜率
            'risk_reward_ratio': round(risk_reward_ratio, 3),  # 盈亏比
            'avg_profit': round(avg_profit, 2),  # 平均盈利
            'avg_loss': round(avg_loss, 2),  # 平均亏损
            'information_ratio': round(information_ratio, 3),  # 信息比率
            'alpha': round(alpha, 3),  # 阿尔法
            'beta': round(beta, 3),  # 贝塔
            'base_annual_return': round(base_annual_return, 3),  # 基准收益率
            'annual_return': round(annual_return, 3),  # 年化收益率
            'cum_returns_final': round(cum_returns_final, 3),  # 累计收益率
            'volatility': round(volatility, 3),  # 收益波动率
            'omega_ratio': round(omega_ratio, 3),  # omega比率
            'calmar_ratio': round(calmar_ratio, 3),  # calmar比率
            'sortino_ratio': round(sortino_ratio, 3),  # sortino比率
            'sharpe_ratio': round(sharpe_ratio, 3),  # sharpe比率
        }

        #画图
        first_day=pd.to_datetime(benchmark_portfolio_daily.index[0]).date()
        print(f'回测第一天{first_day}')
        plot_strategy_data(first_day,benchmark_portfolio_daily,self.plot_out_path)

        # #读取多个基金数据
        # fund_dir="/home/quant/zc/finance_deal/fund_result/Quantitative_Funds"
        # fund1=pd.read_csv(f'{fund_dir}/014806.OF.csv')
        # fund2 = pd.read_csv(f'{fund_dir}/002210.OF.csv')
        # fund3 = pd.read_csv(f'{fund_dir}/001974.OF.csv')
        #
        # #定义基金名称
        # fund_names=['014806.OF','006195.OF','970041.OF']
        # plot_strategy_with_multiple_funds(first_day,benchmark_portfolio_daily,fund_data_list=[fund1,fund2,fund3],fund_names=fund_names)

    
    def notify_trade(self, trade):
        """
        通知交易结果，并根据交易结果更新策略的统计信息
        
        参数:
            trade: 交易对象
        """
        if trade.isclosed:
            if trade.pnlcomm > 0:
                self.wins += 1
                self.gross_profits += trade.pnlcomm
            else:
                self.losses += 1
                self.gross_losses += trade.pnlcomm
    
    def analyze(self, strategy=None):
        """
        分析策略性能
        
        参数:
            strategy: 要分析的策略对象(可选)
            
        返回:
            分析结果字典
        """
        if strategy:
            self.strategy = strategy
            self.stop()
        
        return self.rets.get('datas', {})


class PerformanceAnalyzer(BaseAnalyzer):
    """
    性能分析器
    
    用于离线分析策略性能，不依赖于backtrader框架
    """
    
    def analyze(self, data):
        """
        分析策略性能数据
        
        参数:
            data: 包含策略性能数据的DataFrame
            
        返回:
            分析结果字典
        """
        # 计算收益率
        data['value_daily_ratio'] = data['value'].pct_change()
        data['value_ratio'] = data['value'] / data['value'].iloc[0] - 1
        
        # 计算超额收益
        data['diff'] = data['value_daily_ratio'] - data['daily_return']
        data['cumdiff'] = data['value_ratio'] - data['return']
        
        # 计算各种指标
        mean_excess_return = np.mean(data['diff'])
        std_excess_return = np.std(data['diff'])
        information_ratio = mean_excess_return / std_excess_return if std_excess_return > 0 else 0
        
        alpha = er.stats.alpha(data['value_daily_ratio'], data['daily_return'])
        beta = er.stats.beta(data['value_daily_ratio'], data['daily_return'])
        annual_return = er.stats.annual_return(data['value_daily_ratio'])
        base_annual_return = er.stats.annual_return(data['daily_return'])
        cum_returns_final = er.stats.cum_returns_final(data['value_daily_ratio'])
        volatility = er.stats.annual_volatility(data['value_daily_ratio'])
        sharpe_ratio = er.stats.sharpe_ratio(data['value_daily_ratio'])
        max_drawdown = er.stats.max_drawdown(data['value_daily_ratio'])
        
        # 返回结果
        return {
            'information_ratio': round(information_ratio, 3),
            'alpha': round(alpha, 3),
            'beta': round(beta, 3),
            'annual_return': round(annual_return, 3),
            'base_annual_return': round(base_annual_return, 3),
            'cum_returns_final': round(cum_returns_final, 3),
            'volatility': round(volatility, 3),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(max_drawdown, 3)
        } 