import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any
import logging


@dataclass
class DatabaseConfig:
    """数据库配置"""
    jylh_username: str = 'jylh'
    jylh_password: str = 'jylh'
    jylh_host: str = '10.6.60.114:1521'
    jylh_service: str = 'wind'

    wind_username: str = 'wind'
    wind_password: str = 'wind'
    wind_host: str = '10.6.60.114:1521'
    wind_service: str = 'wind'


@dataclass
class StrategyConfig:
    """策略配置"""
    start_date: str = '2024-10-01'
    end_date: str = '2026-01-01'
    period: int = 1
    execution_price: str = 'open'
    score_threshold: float = -3.0
    topk: int = 100
    low_stopping: float = 1.02
    high_limit: float = 0.98
    vol_percent: float = 0.1
    max_position: float = 0.95
    weighting_method: str = 'equal'  # equal / mktcap
    deal_dividend: bool = False
    cash2shares: bool = True
    save_state: bool = False  #预测:True，回测：False,存储增量回测状态时为True
    month_ibt: bool = False
    adjust_freq: str='M'  #调仓频率：'D'=日频，'M'=月频，'W'=周频，'Q'=季频
    adjust_day_offset: int=1
    initial_stocks: int=50  #初始建仓股票数量
    adjust_count: int=5   # 每次调仓的股票数量
    min_holding_days: int=30  # 最小持有天数,设置为0表示关闭持有时间约束
    stock_pool_mode: int=1  # 股票池模式：0=关闭约束，1=启用约束
    market_cap_threshold: int=50  # 市值阈值(亿元),设置为0表示关闭市值检查
    consecutive_limit_days: int=2 # 连续涨跌停天数限制,设置为0表示关闭约束
    st_stock_mode: int=1 # ST股票模式：0=关闭约束，1=启用约束
    enable_rebalance: bool=True  # 是否启用再平衡功能，True启用，False禁用
    round_to_hundred: bool=True # 是否将股票数量设置为100的整数倍
    is_predict: bool=False  #预测:True，回测：False,存储增量回测状态时为False

@dataclass
class OptimizerConfig:
    """优化器配置"""
    weights_upper: float = 1.0
    weights_offset: float = 0.01
    marketcaps_offset: float = 0.1
    industry_offset: float = 0.01
    turnover_upper: float = 1.0


@dataclass
class BenchmarkConfig:
    """基准配置"""
    indices: Dict[str, str] = field(default_factory=lambda: {
        '沪深300红利全收益指数': '2070003751',
        '沪深300指数': '2070000060',
        '中证A100指数': '2070000103',
        '中证500红利全收益指数': '2070003752',
        '中证小盘500指数': '2070000187',
        '中证800指数': '2070000191',
        '中证1000指数': '2070010507',
        '中证2000指数': '2070023800',
        '食品饮料': '2070008348',
    })
    default: str = '沪深300指数'
    source:str="index_code" #default/index_code/self_stock
    index_instrument:str="000300.SH" #399808.SZ



@dataclass
class BacktestConfig:
    """回测配置"""
    cash: float = 100000000.0
    commission: float = 0.0003
    slippage: float = 0.005

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)



@dataclass
class PathsConfig:
    """路径配置"""
    data: str = './data'
    result: str = './result'
    state: str = './state'
    plot: str = './plot'


class Config:
    """简化的配置管理类

    使用直接属性访问，消除复杂的多级键解析逻辑
    """

    def __init__(self, config_file: str = None):
        """初始化配置

        Args:
            config_file: 配置文件路径
        """
        # 设置默认配置
        self.database = DatabaseConfig()
        self.backtest = BacktestConfig()
        self.paths = PathsConfig()

    def update_strategy_params(self,**kwargs):
        """
        更新策略配置参数
        """
        for key,value in kwargs.items():
            if hasattr(self.backtest.strategy,key):
                #获取原始属性的类型并转换
                original_value=getattr(self.backtest.strategy,key)
                if isinstance(original_value,bool):
                    value=str(value).lower() in 'true'
                elif isinstance(original_value,int):
                    value=int(value)
                elif isinstance(original_value,float):
                    value=float(value)
                elif isinstance(original_value,str):
                    value=str(value)

                setattr(self.backtest.strategy,key,value)
                print(f"Updated strategy.{key}={value}")





# 全局配置实例
config = Config()

