import pandas as pd


class BenchmarkCalculator:
    """
    基准指数计算组件

    负责计算基准指数的收益率
    """

    def __init__(self, benchmark_data, start_date, end_date):
        """
        初始化基准指数计算组件

        参数:
            benchmark_data: 基准指数数据，包含['TRADEDATE', 'TCLOSE']
            start_date: 起始日期
            end_date: 结束日期
        """
        self.benchmark_data = benchmark_data
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_portfolio_daily = None
        self.benchmark_annualReturn = {}

        # 初始化时计算基准指数收益
        self.calculate_benchmark_annual_return()

    def calculate_benchmark_annual_return(self):
        """
        计算基准指数的每年收益
        """
        df = self.benchmark_data[['TRADEDATE', 'TCLOSE']]

        print(f"基准日期：{list(self.benchmark_data['TRADEDATE'])}")

        df.set_index('TRADEDATE', inplace=True)
        # 与回测数据对齐
        df = df[self.start_date: self.end_date]
        # 计算每日收益率
        df['daily_return'] = df['TCLOSE'].pct_change()
        init_close = df['TCLOSE'].iloc[0]
        print(f'init_close为{init_close}')
        df['return'] = df['TCLOSE'] / init_close - 1  # 累计收益率
        self.benchmark_portfolio_daily = df[self.start_date: self.end_date]

        # 提取年份并计算累计收益
        df['year'] = df.index.str[:4].astype(int)  # 提取年份并转换为整数
        df['annual_multiplier'] = (1 + df['daily_return']).fillna(1)  # 每日收益转换为乘积形式

        # 按年份分组计算每年的累计收益
        annual_cumulative = df.groupby('year')['annual_multiplier'].prod()  # 按年份计算乘积
        self.benchmark_annualReturn = (annual_cumulative - 1).to_dict()  # 转为收益率并存为字典

        print(f'基准指数数据每年收益{self.benchmark_annualReturn}')

    def get_benchmark_portfolio_daily(self):
        """
        获取基准投资组合每日数据

        返回:
            基准投资组合每日数据
        """
        return self.benchmark_portfolio_daily

    def get_benchmark_annual_return(self):
        """
        获取基准年化收益

        返回:
            基准年化收益
        """
        return self.benchmark_annualReturn

    def calculate_strategy_performance(self, strategy_values, strategy_dates=None):
        """
        计算策略相对于基准的表现

        参数:
            strategy_values: 策略每日资产价值列表

        返回:
            包含策略和基准表现对比的DataFrame
        """
        # 复制基准数据
        result_df = self.benchmark_portfolio_daily.copy()
        if strategy_dates is not None and len(strategy_values) > 0 and len(strategy_dates) > 0:
            strategy_len = min(len(strategy_values), len(strategy_dates))
            if len(strategy_values) != len(strategy_dates):
                print(
                    f"警告：策略值数量({len(strategy_values)})与策略日期数量({len(strategy_dates)})不一致，按最短长度{strategy_len}对齐")

            strategy_df = pd.DataFrame({
                'TRADEDATE': pd.Series(strategy_dates[:strategy_len]).astype(str),
                'value': strategy_values[:strategy_len]
            })
            strategy_df = strategy_df.drop_duplicates(subset=['TRADEDATE'], keep='last')
            strategy_df = strategy_df.set_index('TRADEDATE')

            benchmark_dates = set(result_df.index.astype(str))
            strategy_date_set = set(strategy_df.index.astype(str))
            benchmark_only = sorted(list(benchmark_dates - strategy_date_set))
            strategy_only = sorted(list(strategy_date_set - benchmark_dates))
            if benchmark_only or strategy_only:
                print(
                    f"警告：策略日期与基准日期存在差异，"
                    f"仅基准日期数={len(benchmark_only)}，仅策略日期数={len(strategy_only)}"
                )
                if benchmark_only:
                    print(f"仅基准日期样例:{benchmark_only[:5]}")
                if strategy_only:
                    print(f"仅策略日期样例:{strategy_only[:5]}")

            result_df['value'] = strategy_df['value'].reindex(result_df.index.astype(str))
            result_df['value'] = result_df['value'].ffill().bfill()
        else:
            if len(strategy_values) != len(result_df):
                print(f"警告：策略值长度({len(strategy_values)})与基准数据长度({len(result_df)})不匹配")
                if len(strategy_values) < len(result_df) and len(strategy_values) > 0:
                    strategy_values = strategy_values + [strategy_values[-1]] * (len(result_df) - len(strategy_values))
                elif len(strategy_values) > len(result_df):
                    strategy_values = strategy_values[:len(result_df)]
            result_df['value'] = strategy_values

        # 计算策略每日收益率
        result_df['value_daily_ratio'] = result_df['value'].pct_change()

        # 计算策略累计收益率
        initial_value = result_df['value'].iloc[0]
        result_df['value_ratio'] = result_df['value'] / initial_value - 1

        # 计算策略与基准的每日收益差
        result_df['diff'] = result_df['value_daily_ratio'] - result_df['daily_return']

        # 计算策略与基准的累计收益差
        result_df['cumdiff'] = result_df['value_ratio'] - result_df['return']

        return result_df 
