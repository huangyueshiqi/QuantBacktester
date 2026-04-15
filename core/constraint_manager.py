"""约束管理器

这个模块实现了交易约束的统一管理
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, Tuple
import pandas as pd


class BaseConstraint(ABC):
    """约束基类
    """

    @abstractmethod
    def check(self, stock_code: str, context: Dict[str, Any]) -> bool:
        """检查股票是否满足约束条件

        Args:
            stock_code: 股票代码
            context: 上下文信息，包含策略实例、当前日期等

        Returns:
            True表示满足约束（可以交易），False表示违反约束
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取约束名称"""
        pass


class MinHoldingDaysConstraint(BaseConstraint):
    """最小持有天数约束"""

    def __init__(self, min_days: int):
        self.min_days = min_days
        self.enabled = min_days > 0

    def check(self, stock_code: str, context: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        #通过context获取持有天数数据
        strategy = context.get('strategy')
        current_date=context.get('current_date')
        holding_days=strategy.portfolio.get_holding_days(stock_code,current_date)

        return holding_days>=self.min_days

    def get_name(self) -> str:
        return f"最小持有期{self.min_days}天约束"


class StockPoolConstraint(BaseConstraint):
    """股票池约束"""

    def __init__(self, stock_pool: Set[str], mode: int):
        self.stock_pool = stock_pool
        self.mode = mode
        self.enabled = mode > 0 and bool(stock_pool)

    def check(self, stock_code: str, context: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        return stock_code in self.stock_pool

    def get_name(self) -> str:
        status = "启用" if self.enabled else "禁用"
        return f"股票池约束({status})"


class STStockConstraint(BaseConstraint):
    """ST股票约束"""
    def __init__(self,mode:int):
        self.mode=mode
        self.enabled = mode > 0

    def check(self, stock_code: str, context: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        st_list = context.get('st_list',[])
        return stock_code not in st_list

    def get_name(self) -> str:
        status = "启用" if self.enabled else "禁用"
        return f"ST股票约束({status})"



class ConsecutiveLimitConstraint(BaseConstraint):
    """连续涨跌停约束"""

    def __init__(self, consecutive_days: int,high_limit,low_stopping):
        self.consecutive_days = consecutive_days
        self.high_limit=high_limit
        self.low_stopping=low_stopping
        self.enabled = consecutive_days > 0

    def check(self, stock_code: str, context: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        # 通过context获取股票数据
        strategy = context.get('strategy')
        stock_data=strategy.getdatabyname(stock_code)

        if stock_data is None or len(stock_data) < self.consecutive_days + 1:
            return True

        # 检查连续涨停
        consecutive_up_limit = self._check_consecutive_up_limit(stock_data)
        if consecutive_up_limit >= self.consecutive_days:
            return False

        # 检查连续跌停
        consecutive_down_limit = self._check_consecutive_down_limit(stock_data)
        if consecutive_down_limit >= self.consecutive_days:
            return False

        return True

    def _check_consecutive_up_limit(self,stock_data):
        """检查连续涨停天数"""
        consecutive_count = 0
        # 检查过去天的数据(不包括当天，从-1开始往前看)
        for i in range(1, self.consecutive_days + 1):
            # 当最高价超过涨停价的98%时认为涨停
            if stock_data.high[-i] >= stock_data.limit[-i] * self.high_limit:
                consecutive_count += 1
            else:
                break
        return consecutive_count

    def _check_consecutive_down_limit(self,stock_data):
        """检查连续跌停天数"""
        consecutive_count = 0
        # 检查过去天的数据(不包括当天，从-1开始往前看)
        for i in range(1, self.consecutive_days + 1):
            # 当最低价低于跌停价的102%时认为跌停
            if stock_data.low[-i] <= stock_data.stopping[-i] * self.low_stopping:
                consecutive_count += 1
            else:
                break
        return consecutive_count



    def get_name(self) -> str:
        if self.enabled:
            return f"连续{self.consecutive_days}天涨跌停约束"
        return "连续涨跌停约束(禁用)"


class ConstraintManager:
    """约束管理器
    """

    def __init__(self):
        self.constraints: List[BaseConstraint] = []
        self.statistics = {}  # 统计各约束的过滤情况

    def add_constraint(self, constraint: BaseConstraint):
        """添加约束"""
        self.constraints.append(constraint)

    def _filter_constraints_by_type(self,constraint_types):
        """
        根据约束类型名称过滤约束
        """
        type_mapping={
            'MinHoldingDays':MinHoldingDaysConstraint,
            'StockPool':StockPoolConstraint,
            'ConsecutiveLimit':ConsecutiveLimitConstraint,
            'STStock':STStockConstraint
        }

        filter_constraints=[]
        for constraint in self.constraints:
            for type_name in constraint_types:
                if type_name in type_mapping and isinstance(constraint,type_mapping[type_name]):
                    filter_constraints.append(constraint)
                    break

        return filter_constraints


    def check_stock(self, stock_code: str, context: Dict[str, Any],constraint_types=None) -> Tuple[bool, List[str]]:
        """检查单只股票是否满足所有约束

        Args:
            stock_code: 股票代码
            context: 上下文信息
            constraint_types:要检查的约束类型列表，None表示检查所有约束

        Returns:
            (是否通过所有约束, 违反的约束名称列表)
        """
        violated_constraints = []

        #如果没有指定约束类型，则检查所有约束
        if constraint_types is None:
            constraints_to_check=self.constraints
        else:
            #根据类型名称过滤约束
            constraints_to_check=self._filter_constraints_by_type(constraint_types)

        for constraint in constraints_to_check:
            if not constraint.check(stock_code, context):
                violated_constraints.append(constraint.get_name())

        return len(violated_constraints) == 0, violated_constraints


    def filter_stocks(self, stock_list: List[str], context: Dict[str, Any],
                      operation_type: str = "交易",constraint_types=None) -> Tuple[List[str], Dict[str, List[str]]]:
        """批量过滤股票

        Args:
            stock_list: 股票列表
            context: 上下文信息
            operation_type: 操作类型，用于日志

        Returns:
            (通过约束的股票列表, 被过滤股票的约束违反信息)
        """
        passed_stocks = []
        filtered_info = {}

        for stock in stock_list:
            passed, violated = self.check_stock(stock, context,constraint_types)
            if passed:
                passed_stocks.append(stock)
            else:
                filtered_info[stock] = violated
                # 更新统计信息
                for constraint_name in violated:
                    if constraint_name not in self.statistics:
                        self.statistics[constraint_name] = 0
                    self.statistics[constraint_name] += 1

        # 统计过滤信息
        self._log_filter_results(stock_list, passed_stocks, filtered_info, operation_type, context)

        return passed_stocks, filtered_info

    def _log_filter_results(self, original_list: List[str], passed_list: List[str],
                            filtered_info: Dict[str, List[str]], operation_type: str,
                            context: Dict[str, Any]):
        """
        记录过滤结果

        这里做了一个数据重组：
        - 原来的数据： {"股票"：["违规约束1","违规约束2"]}
        - 重组的数据： {"违规约束1"：["股票A","股票B"],"违规约束2"：["股票C"]}
        """
        strategy = context.get('strategy')
        if not strategy:
            return

        filtered_count = len(original_list) - len(passed_list)
        if filtered_count > 0:
            strategy.log(f"{operation_type}约束过滤：过滤掉{filtered_count}只股票")

            # 按约束类型统计
            constraint_stats = {}
            for stock, violations in filtered_info.items():
                for violation in violations:
                    if violation not in constraint_stats:
                        constraint_stats[violation] = []
                    constraint_stats[violation].append(stock)

            # 输出详细统计
            for constraint_name, stocks in constraint_stats.items():
                strategy.log(f"  {constraint_name}: {len(stocks)}只 {stocks[:5]}{'...' if len(stocks) > 5 else ''}")

    def get_constraint_status(self) -> str:
        """获取所有约束的状态描述"""
        if not self.constraints:
            return "无约束"

        status_list = [constraint.get_name() for constraint in self.constraints]
        return "; ".join(status_list)

    def get_statistics(self):
        """获取约束过滤统计信息"""
        return self.statistics.copy()

    def reset_statistics(self):
        """重置统计信息"""
        self.statistics = {}

    def check_single_constraint(self,stock_code,context,constraint_type):
        """检查单个约束"""
        passed,violated=self.check_stock(stock_code,context,[constraint_type])
        return passed,violated if violated else None

    def filter_for_buying(self,stock_list,context):
        """买入专用过滤：检查股票池和连续涨跌停约束"""
        filtered_stocks,_ =self.filter_stocks(
            stock_list,context,'买入检查',
            constraint_types=['ConsecutiveLimit','STStock','StockPool']
        )
        return filtered_stocks


def create_constraint_manager_from_params(params, stock_pool: Set[str]) -> ConstraintManager:
    """从策略参数创建约束管理器

    这个工厂函数消除了在策略类中散布的约束初始化逻辑
    """
    manager = ConstraintManager()

    # 添加各种约束
    manager.add_constraint(MinHoldingDaysConstraint(params.min_holding_days))
    manager.add_constraint(StockPoolConstraint(stock_pool, params.stock_pool_mode))
    manager.add_constraint(ConsecutiveLimitConstraint(
        getattr(params, 'consecutive_limit_days', 0),
        getattr(params, 'high_limit', 0.98),
        getattr(params, 'low_stopping', 1.02),
    ))
    manager.add_constraint(STStockConstraint(params.st_stock_mode))

    return manager