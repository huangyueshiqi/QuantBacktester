from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class Position:
    """
    单个持仓记录
    统一管理股票的买入信息、数量和分红现金
    """

    def __init__(self, stock_code: str, buy_date: str, size: float, price: float):
        self.stock_code = stock_code
        self.buy_date = buy_date  # 字符串格式 'YYYY-MM-DD'
        self.size = size
        self.price = price
        self.dividend_cash = 0.0  # 该持仓产生的分红现金

    def can_sell(self, current_date: str, min_holding_days: int) -> bool:
        """
        检查是否满足最小持有期要求

        参数:
            current_date: 当前日期字符串 'YYYY-MM-DD'
            min_holding_days: 最小持有天数

        返回:
            bool: True表示可以卖出，False表示需要继续持有
        """
        if min_holding_days <= 0:
            return True

        current_dt = pd.to_datetime(current_date)
        buy_dt = pd.to_datetime(self.buy_date)
        holding_days = (current_dt - buy_dt).days

        return holding_days >= min_holding_days

    def add_dividend_cash(self, amount: float):
        """添加分红现金"""
        self.dividend_cash += amount

    def get_market_value(self, current_price: float) -> float:
        """获取当前市值"""
        return self.size * current_price

    def __repr__(self):
        return f"Position({self.stock_code}, {self.buy_date}, {self.size}, {self.price})"


class Portfolio:
    """
    投资组合管理器
    统一管理所有持仓，替代原来的四个分散数据结构：
    - FIFO_positions
    - FIFO_pos2cash
    - stock_buy_dates
    - current_holding
    """

    def __init__(self):
        # 使用字典存储持仓，支持FIFO逻辑
        # {stock_code: {buy_date: Position}}
        self._positions: Dict[str, Dict[str, Position]] = {}

    def add_position(self, stock_code: str, buy_date: str, size: float, price: float):
        """
        添加持仓记录

        参数:
            stock_code: 股票代码
            buy_date: 买入日期 'YYYY-MM-DD'
            size: 买入数量
            price: 买入价格
        """
        if stock_code not in self._positions:
            self._positions[stock_code] = {}

        if buy_date not in self._positions[stock_code]:
            self._positions[stock_code][buy_date] = Position(stock_code, buy_date, 0, price)

        # 累加同一天的买入数量
        self._positions[stock_code][buy_date].size += size

    def remove_position(self, stock_code: str, size_to_sell: float) -> List[Position]:
        """
        按FIFO原则卖出持仓

        参数:
            stock_code: 股票代码
            size_to_sell: 要卖出的数量

        返回:
            List[Position]: 被卖出的持仓记录列表
        """
        if stock_code not in self._positions:
            return []

        sold_positions = []
        remaining_to_sell = abs(size_to_sell)

        # 按买入日期排序，先进先出
        sorted_dates = sorted(self._positions[stock_code].keys())

        for buy_date in sorted_dates:

            position = self._positions[stock_code][buy_date]

            if position.size <= remaining_to_sell:
                # 全部卖出这个持仓
                sold_positions.append(position)
                remaining_to_sell -= position.size
                del self._positions[stock_code][buy_date]
            else:
                # 部分卖出
                sold_position = Position(stock_code, buy_date, remaining_to_sell, position.price)
                sold_position.dividend_cash = position.dividend_cash * (remaining_to_sell / position.size)
                sold_positions.append(sold_position)

                # 更新剩余持仓
                position.size -= remaining_to_sell
                position.dividend_cash -= sold_position.dividend_cash
                remaining_to_sell = 0

                #清理极小持仓(避免浮点精度问题)
                if position.size<1e-2:
                    del self._positions[stock_code][buy_date]

        # 如果股票完全卖出，删除该股票记录
        if stock_code in self._positions and not self._positions[stock_code]:
            del self._positions[stock_code]

        return sold_positions

    def get_holding_stocks(self) -> set:
        """获取当前持仓股票集合"""
        return set(self._positions.keys())

    def get_stock_buy_date(self, stock_code: str) -> Optional[str]:
        """获取股票最早买入日期"""
        if stock_code not in self._positions:
            return None

        return min(self._positions[stock_code].keys())

    def get_stock_total_size(self, stock_code: str) -> float:
        """获取股票总持仓数量"""
        if stock_code not in self._positions:
            return 0.0

        return sum(pos.size for pos in self._positions[stock_code].values())

    def get_stock_positions(self, stock_code: str) -> Dict[str, Position]:
        """获取某只股票的所有持仓记录"""
        return self._positions.get(stock_code, {})

    def get_all_positions(self) -> Dict[str, Dict[str, Position]]:
        """获取所有持仓记录"""
        return self._positions.copy()

    def add_dividend_cash(self, stock_code: str, total_dividend: float):
        """
        为某只股票的所有持仓按比例分配分红现金

        参数:
            stock_code: 股票代码
            total_dividend: 总分红金额
        """
        if stock_code not in self._positions:
            return

        total_size = self.get_stock_total_size(stock_code)
        if total_size <= 0:
            return

        # 按持仓比例分配分红
        for position in self._positions[stock_code].values():
            dividend_per_position = total_dividend * (position.size / total_size)
            position.add_dividend_cash(dividend_per_position)

    def get_total_dividend_cash(self, stock_code: str) -> float:
        """获取某只股票的总分红现金"""
        if stock_code not in self._positions:
            return 0.0

        return sum(pos.dividend_cash for pos in self._positions[stock_code].values())

    def can_sell_stock(self, stock_code: str, current_date: str, min_holding_days: int) -> bool:
        """
        检查股票是否满足卖出条件（最小持有期要求）

        参数:
            stock_code: 股票代码
            current_date: 当前日期字符串 'YYYY-MM-DD'
            min_holding_days: 最小持有天数

        返回:
            bool: True表示可以卖出，False表示需要继续持有
        """
        # 检查最早的持仓是否满足持有期要求
        earliest_date = self.get_stock_buy_date(stock_code)

        earliest_position = self._positions[stock_code][earliest_date]
        return earliest_position.can_sell(current_date, min_holding_days)

    def get_holding_days(self,stock_code: str, current_date: str) -> int:
        """
        计算股票持有天数
        """
        # 检查最早的持仓是否满足持有期要求
        buy_date = self.get_stock_buy_date(stock_code)
        if buy_date:
            current_dt = pd.to_datetime(current_date)
            buy_dt = pd.to_datetime(buy_date)
            return (current_dt - buy_dt).days
        else:
            return 0

    def add_dividend_cash_by_date(self, stock, buy_date, dividend_amount):
        """
        为指定股票的指定买入日期添加分红现金

        参数:
            stock: 股票代码
            buy_date: 买入日期
            dividend_amount: 分红金额
        """
        if stock in self._positions and buy_date in self._positions[stock]:
            self._positions[stock][buy_date].dividend_cash += dividend_amount

    def to_dict(self):
        """
        将Portfolio对象转换为字典格式，用于序列化

        返回:
            包含所有持仓信息的字典
        """
        result = {}
        for stock_code, positions in self._positions.items():
            result[stock_code] = {
                buy_date: {
                    'stock_code': pos.stock_code,
                    'buy_date': pos.buy_date,
                    'size': pos.size,
                    'price': pos.price,
                    'dividend_cash': pos.dividend_cash
                }
                for buy_date, pos in positions.items()
            }
        return result

    def from_dict(self, data):
        """
        从字典格式恢复Portfolio对象，用于反序列化

        参数:
            data: 包含持仓信息的字典
        """
        self._positions.clear()
        for stock_code, positions_data in data.items():
            self._positions[stock_code] = {}
            for buy_date, pos_data in positions_data.items():
                position = Position(
                    stock_code=pos_data['stock_code'],
                    buy_date=pos_data['buy_date'],
                    size=pos_data['size'],
                    price=pos_data['price']
                )
                position.dividend_cash = pos_data.get('dividend_cash', 0.0)
                self._positions[stock_code][buy_date] = position

    def is_empty(self) -> bool:
        """检查投资组合是否为空"""
        return len(self._positions) == 0

    def get_portfolio_summary(self) -> Dict:
        """
        获取投资组合摘要信息

        返回:
            Dict: 包含持仓统计信息
        """
        summary = {
            'total_stocks': len(self._positions),
            'stocks': list(self._positions.keys()),
            'total_positions': sum(len(positions) for positions in self._positions.values()),
            'total_dividend_cash': sum(
                sum(pos.dividend_cash for pos in positions.values())
                for positions in self._positions.values()
            )
        }
        return summary

    def __repr__(self):
        return f"Portfolio({len(self._positions)} stocks, {sum(len(p) for p in self._positions.values())} positions, positions:{self._positions})"