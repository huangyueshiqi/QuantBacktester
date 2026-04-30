import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib
import time
import pickle
from utils import empyrical_compat as er
from matplotlib import font_manager

from utils.config import config
from data.loader import OracleDataLoader
from core.portfolio import CvxPortfolioOptimizer

# 配置中文字体支持
# font_path = '/usr/share/fonts/cjkuni-uming/uming.ttc'  # 服务器可用字体路径
font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')   # 设置后端

# -------------- 日期时间处理函数 ----------------

def calculate_holding_period(buy_date, sell_date):
    """
    计算持股月份数量
    
    参数:
        buy_date: 买入日期
        sell_date: 卖出日期
        
    返回:
        持股月数
    """
    buy_datetime = datetime.strptime(buy_date, '%Y-%m-%d')
    sell_datetime = datetime.strptime(sell_date, '%Y-%m-%d')
    delta = relativedelta(sell_datetime, buy_datetime)
    holding_months = delta.years * 12 + delta.months
    return holding_months


def calculate_tax(buy_date, sell_date):
    """
    持股周期对应的扣税系数
    
    参数:
        buy_date: 购买日期
        sell_date: 卖出日期
        
    返回:
        扣税系数
    """
    holding_months = calculate_holding_period(buy_date, sell_date)
    if holding_months <= 1:
        return 0.2
    elif holding_months <= 12:
        return 0.1
    else:
        return 0


def rebalancing_day(start_date, end_date, freq='ME', n_business_days=1):
    """
    获取调仓日期列表
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        freq: 调仓频率（默认每月初）,支持'W'(周),'M'(月),'Q'(季度),'Y'(年度)
        n_business_days: 取每月第N个交易日
        
    返回:
        调仓日期列表
    """
    # 将日期字符串转换为日期对象
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 获取交易日历数据
    loader = OracleDataLoader()
    df = loader.load_data('calendar')
    
    # 删除交易日中包含NaN的行
    clean_trading_days = df.dropna(subset=['TRADE_DAYS'])
    # 删除除交易日中重复项
    clean_trading_days = clean_trading_days.drop_duplicates(subset=['TRADE_DAYS']).reset_index(drop=True)
    # 将交易日列转换为日期类型
    clean_trading_days['TRADE_DAYS'] = pd.to_datetime(clean_trading_days['TRADE_DAYS'])
    
    # 筛选时间范围内的交易日
    mask = (clean_trading_days['TRADE_DAYS'] >= start_dt) & (clean_trading_days['TRADE_DAYS'] <= end_dt)
    filtered_days = clean_trading_days['TRADE_DAYS'][mask]
    
    # 生成调仓日
    rebalance_dates = (filtered_days.to_frame().groupby(pd.Grouper(key='TRADE_DAYS', freq=freq)).nth(n_business_days - 1))
    # 将结果转换为列表并返回
    rebalance_dates = rebalance_dates['TRADE_DAYS'].dt.strftime('%Y-%m-%d').tolist()
    
    return rebalance_dates


def rebalancing_by_period(start_date, end_date, period=20):
    """
    根据交易日周期获取调仓日期列表

    参数:
        start_date: 开始日期
        end_date: 结束日期
        period: 调仓周期（交易日数量）

    返回:
        调仓日期列表
    """
    # 将日期字符串转换为日期对象
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 获取交易日历数据
    loader = OracleDataLoader()
    df = loader.load_data('calendar')

    # 删除交易日中包含NaN的行
    df_no = df.dropna(subset=['TRADE_DAYS'])
    # 删除除交易日中重复项
    df_no = df_no.drop_duplicates(subset=['TRADE_DAYS']).reset_index(drop=True)
    # 将交易日列转换为日期类型
    df_no['TRADE_DAYS'] = pd.to_datetime(df_no['TRADE_DAYS'])

    # 筛选时间范围内的交易日
    mask = (df_no['TRADE_DAYS'] >= start_dt) & (df_no['TRADE_DAYS'] <= end_dt)
    filtered_days = df_no['TRADE_DAYS'][mask]

    # 按照指定的交易日周期生成调仓日
    rebalance_indices = list(range(0,len(filtered_days),period))
    rebalance_dates = filtered_days.iloc[rebalance_indices].dt.strftime('%Y-%m-%d').tolist()

    return rebalance_dates


def get_trade_days():
    """
    获取交易日数据并进行预处理
    
    返回:
        交易日数据DataFrame
    """
    loader = OracleDataLoader()
    df = loader.load_data('calendar')
    
    # 删除交易日中包含NaN的行
    df_no_nan = df.dropna(subset=['TRADE_DAYS'])
    # 删除交易日中重复项
    df_no_duplicate = df_no_nan.drop_duplicates(subset=['TRADE_DAYS']).reset_index(drop=True)
    return df_no_duplicate


def get_previous_trading_day(trade_days, target_day):
    """
    找到目标调仓日
    
    参数:
        trade_days: 交易日列表
        target_day: 目标日期
        
    返回:
        目标调仓日
    """
    # 找到目标调仓日
    target_day = pd.to_datetime(target_day)
    prev_day = trade_days[trade_days == target_day]
    prev_day = pd.Timestamp(prev_day.values[0])
    return prev_day


def bt_periods(monthly_rebalance_dates, start_date, end_date):
    """
    根据调仓日列表获取增量回测周期
    
    参数:
        monthly_rebalance_dates: 调仓日列表
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        增量回测周期列表
    """
    # 划分每个回测周期的开始和结束日期
    periods = []
    # 获取交易日数据
    df_no = get_trade_days()
    trade_days = pd.to_datetime(df_no['TRADE_DAYS'])
    
    # 输入日期转换为pd.Timestamp
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    monthly_rebalance_dates = pd.to_datetime(monthly_rebalance_dates)
    
    # 筛选出时间范围内的调仓日
    trade_days = trade_days[(trade_days >= start_date) & (trade_days <= end_date)]
    
    for i in range(1, len(monthly_rebalance_dates)):
        if i == 1:
            start = monthly_rebalance_dates[i - 1]
        else:
            start = end
        end = get_previous_trading_day(trade_days, monthly_rebalance_dates[i])
        periods.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    
    # 如果最后一个调仓日期在结束日期之前，添加一个周期
    if monthly_rebalance_dates[-1] <= end_date:
        periods.append((end.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    
    return periods


def load_ibt_date(date):
    """
    加载增量回测日期
    
    参数:
        date: 日期
        
    返回:
        前一个交易日
    """
    df_no = get_trade_days()
    trade_days = pd.to_datetime(df_no['TRADE_DAYS']).tolist()
    first_backtest = pd.to_datetime(date)
    prev_index = trade_days.index(first_backtest) - 2    #假设提前两天
    prev_trade = trade_days[prev_index].strftime('%Y-%m-%d')
    return prev_trade


# -------------- 数据处理函数 ----------------

def get_stock_dic(datas):
    """
    回测中遍历数据源并提取股票信息字典
    
    参数:
        datas: 数据源
        
    返回:
        股票信息字典
    """
    stock_dic = {}
    for i, data in enumerate(datas):
        stock_name = data._name  # 股票名
        score = data.score[0]
        marketvalue = data.MARKETVALUE[0]
        indecode = data.indecode[0]
        price = data.close[0]
        
        # 将数据存入字典
        stock_dic[stock_name] = {
            'score': score,
            'marketvalue': marketvalue,
            'indecode': indecode,
            'price': price,
        }
    
    return stock_dic


def get_index_dic(datas):
    """
    回测中遍历数据源并提取指数信息字典

    参数:
        datas: 数据源

    返回:
        指数信息字典
    """
    index_dic = {}
    for i, data in enumerate(datas):
        index_name = data._name  # 股票名
        price = data.close[0]

        # 将数据存入字典
        index_dic[index_name] = {
            'price': price,
        }

    return index_dic


def get_stock_dic_from_dataframe(df, cur_date):
    """
    从股票数据dataframe中获取股票数据并转为字典
    
    参数:
        df: 股票数据DataFrame
        cur_date: 当前日期
        
    返回:
        股票信息字典
    """
    # 筛选调仓日期数据
    filtered_df = df[df['datetime'] == cur_date]
    
    # 转换为字典
    stock_dic = filtered_df[['score', 'MARKETVALUE', 'indecode', 'close']].rename(
        columns={'MARKETVALUE': 'marketvalue', 'close': 'price'}).to_dict(orient='index')
    
    return stock_dic


def get_st_stock(st_dict, cur_date):
    """
    找到截至当前日期的最近日期的ST股票列表
    
    参数:
        st_dict: ST股票字典
        cur_date: 当前日期
        
    返回:
        ST股票列表
    """
    cur_st_list = []
    for day in list(reversed(st_dict.keys())):
        if day <= cur_date:
            cur_st_list = st_dict[day]
            break
    return cur_st_list


# -------------- 可视化函数 ----------------

def plot_strategy_data(start_date, plot_df,output_path):
    """
    绘制策略回测结果图
    
    参数:
        start_date: 开始日期
        plot_df: 绘图数据
    """
    if 'date' not in plot_df.columns:
        plot_df=plot_df.reset_index()
        plot_df=plot_df.rename(columns={'TRADEDATE':'date'})
    plot_data = plot_df[['date', 'return', 'value_ratio', 'cumdiff']]
    # 找到value_ratio列中第一个非零元素的位置
    # first_non_zero_index = plot_data['value_ratio'].ne(0).idxmax()
    first_non_zero_index = plot_data['value_ratio'].first_valid_index()
    # 从第一个非零元素开始切片,画图从开始日期开始
    # plot_data.iloc[first_non_zero_index - 1] = 0
    # plot_data = plot_data.loc[first_non_zero_index - 1:]
    # 确保不会越界（比如 first_non_zero_index == 0）

    if first_non_zero_index > 0:
        # 只重置数值列，保留 date 不变
        plot_data.loc[first_non_zero_index - 1, ['return', 'value_ratio', 'cumdiff']] = 0.0
        plot_data = plot_data.loc[first_non_zero_index - 1:]
    else:
        plot_data = plot_data.loc[first_non_zero_index:]
    
    # 设置日期为索引
    plot_data.loc[0, 'date'] = start_date
    plot_data['date'] = pd.to_datetime(plot_data['date'],format="%Y-%m-%d")
    plot_data.set_index('date', inplace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['return'], label='基准收益率')
    plt.plot(plot_data['value_ratio'], label='策略收益率')
    plt.plot(plot_data['cumdiff'], label='超额收益率')
    plt.legend()
    plt.title('策略回测结果图')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    
    # 设置x轴格式为"XXXX年XX月"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
    # 设置x轴只显示每季度的第一天
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    # 自动旋转日期标签，防止重叠
    plt.gcf().autofmt_xdate()
    
    plt.show()
    plt.savefig(output_path,dpi=300,bbox_inches='tight')
    plt.close()
    # plt.savefig('output.png')  # 保存图像到文件
    # print("图形已保存为 output.png")


def plot_strategy_with_multiple_funds(start_date, plot_df, fund_data_list=None, fund_names=None,output_path="plot/strategy_plot.png"):
    """
    绘制策略回测结果图，可加入多个基金收益率对比

    参数:
        start_date: 开始日期
        plot_df: 绘图数据
        fund_data_list: 基金数据列表 (可选)，每个元素应包含'date'和'F_NAV_ADJUSTED'列
        fund_names: 基金名称列表 (可选)，与fund_data_list一一对应
    """
    # 准备基础数据
    if 'date' not in plot_df.columns:
        plot_df = plot_df.reset_index()
        plot_df = plot_df.rename(columns={'TRADEDATE': 'date'})

    plot_data = plot_df[['date', 'return', 'value_ratio', 'cumdiff']].copy()

    # 找到value_ratio列中第一个非零元素的位置
    first_non_zero_index = plot_data['value_ratio'].ne(0).idxmax()
    # 从第一个非零元素开始切片,画图从开始日期开始
    plot_data.iloc[first_non_zero_index - 1] = 0
    plot_data = plot_data.loc[first_non_zero_index - 1:]

    # 设置日期为索引
    plot_data.loc[0, 'date'] = start_date
    plot_data['date'] = pd.to_datetime(plot_data['date'], format="%Y-%m-%d")
    plot_data.set_index('date', inplace=True)

    # 计算多个基金累计收益率（如果提供了基金数据）
    if fund_data_list is not None:
        if fund_names is None:
            fund_names = [f'基金{i + 1}' for i in range(len(fund_data_list))]
        elif len(fund_names) != len(fund_data_list):
            raise ValueError("fund_names的长度必须与fund_data_list相同")

        for i, fund_df in enumerate(fund_data_list):
            fund_data = fund_df.copy()
            fund_data['date'] = pd.to_datetime(fund_data['ANN_DATE'], format="%Y%m%d")
            fund_data.set_index('date', inplace=True)

            # 合并数据
            plot_data = plot_data.join(fund_data['F_NAV_ADJUSTED'], how='left')
            #前向填充
            plot_data['F_NAV_ADJUSTED']=plot_data['F_NAV_ADJUSTED'].fillna(method='ffill')
            # 计算累计收益率
            col_name = f'fund_{i}_cum_return'
            plot_data[col_name] = (plot_data['F_NAV_ADJUSTED'] - plot_data['F_NAV_ADJUSTED'].iloc[0]) / \
                                    plot_data['F_NAV_ADJUSTED'].iloc[0]
            plot_data.drop(columns=['F_NAV_ADJUSTED'],inplace=True)


    plt.figure(figsize=(14, 7))
    # 绘制策略相关曲线
    plt.plot(plot_data['return'], label='基准收益率', linewidth=2)
    plt.plot(plot_data['value_ratio'], label='策略收益率', linewidth=2)
    plt.plot(plot_data['cumdiff'], label='超额收益率', linewidth=2)

    # 绘制基金曲线（如果有）
    if fund_data_list is not None:

        for i in range(len(fund_data_list)):
            col_name = f'fund_{i}_cum_return'
            plt.plot(plot_data[col_name],
                     label=fund_names[i],
                     linewidth=1.5,alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('策略与多基金对比图')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 设置x轴格式为"XXXX年XX月"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # 设置x轴只显示每季度的第一天
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    # 自动旋转日期标签，防止重叠
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()

# -------------- 结果分析函数 ----------------

def load_files_by_dates(dates_list):
    """
    批量加载增量回测中策略的市值csv并合并为一个dataframe
    
    参数:
        dates_list: 日期列表
        
    返回:
        合并后的DataFrame
    """
    combined_df = pd.DataFrame()
    path=config.paths.result
    for i, date in enumerate(dates_list):
        file_path = f'{path}/plot_{date}.csv'
        temp_df = pd.read_csv(file_path)
        # 增量回测中恢复状态需要两天
        if i != 0:
            temp_df = temp_df.iloc[3:]
        combined_df = pd.concat([combined_df, temp_df['value']], ignore_index=True)
    
    combined_df.columns = ['value']
    return combined_df


def monthly_analysis(data):
    """
    计算每月的收益差
    
    参数:
        data: 包含收益率数据的DataFrame
        
    返回:
        每月收益差列表
    """
    # 提取日期和月份
    months = data['date_copy'].dt.month  # 转换为月份
    
    # 初始化存储结果
    monthly_return_diffs = []
    
    # 获取每日收益率和基准收益率
    daily_returns = data['value_daily_ratio'].values
    daily_benchmark_returns = data['daily_return'].values
    
    # 按月份分组计算
    unique_months = np.unique(months)
    for month in unique_months:
        # 获取当前月份的索引
        month_mask = (months == month)
        valid_daily_returns = daily_returns[month_mask]
        valid_benchmark_returns = daily_benchmark_returns[month_mask]
        # 过滤掉NaN值
        valid_daily_returns = valid_daily_returns[~np.isnan(valid_daily_returns)]
        valid_benchmark_returns = valid_benchmark_returns[~np.isnan(valid_benchmark_returns)]
        # 计算收益率和差异
        monthly_return = np.prod(1 + valid_daily_returns) - 1
        monthly_benchmark_return = np.prod(1 + valid_benchmark_returns) - 1
        monthly_return_diff = monthly_return - monthly_benchmark_return
        monthly_return_diffs.append(monthly_return_diff)
    
    return monthly_return_diffs


def yearly_analysis(data):
    """
    计算每年的收益差
    
    参数:
        data: 包含收益率数据的DataFrame
        
    返回:
        每年收益差列表
    """
    # 提取日期和年份
    years = data['date_copy'].dt.year  # 转换为年份
    
    # 初始化存储结果
    yearly_return_diffs = []
    
    # 获取每日收益率和基准收益率
    daily_returns = data['value_daily_ratio'].values
    daily_benchmark_returns = data['daily_return'].values
    
    # 按年份分组计算
    unique_years = np.unique(years)
    for year in unique_years:
        # 获取当前年份的索引
        year_mask = (years == year)
        valid_daily_returns = daily_returns[year_mask]
        valid_benchmark_returns = daily_benchmark_returns[year_mask]
        # 过滤掉NaN值
        valid_daily_returns = valid_daily_returns[~np.isnan(valid_daily_returns)]
        valid_benchmark_returns = valid_benchmark_returns[~np.isnan(valid_benchmark_returns)]
        # 计算收益率和差异
        yearly_return = np.prod(1 + valid_daily_returns) - 1
        yearly_benchmark_return = np.prod(1 + valid_benchmark_returns) - 1
        yearly_return_diff = yearly_return - yearly_benchmark_return
        yearly_return_diffs.append(yearly_return_diff)
    
    return yearly_return_diffs


def calculate_benchmark_annual_return(benchmark_detail_data, start_date, end_date):
    """
    计算基准指数的每年收益
    
    参数:
        benchmark_detail_data: 基准指数数据
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        基准指数每日数据, 基准年化收益
    """
    df = benchmark_detail_data[['TRADEDATE', 'TCLOSE']]
    
    df.set_index('TRADEDATE', inplace=True)
    # 与回测数据对齐
    df = df[start_date: end_date]
    # 计算每日收益率
    df['daily_return'] = df['TCLOSE'].pct_change()
    init_close = df['TCLOSE'].iloc[0]
    df['return'] = df['TCLOSE'] / init_close - 1  # 累计收益率
    benchmark_portfolio_daily = df[start_date: end_date]
    
    # 提取年份并计算累计收益
    df['year'] = df.index.str[:4].astype(int)  # 提取年份并转换为整数
    df['annual_multiplier'] = (1 + df['daily_return']).fillna(1)  # 每日收益转换为乘积形式
    
    # 按年份分组计算每年的累计收益
    annual_cumulative = df.groupby('year')['annual_multiplier'].prod()  # 按年份计算乘积
    benchmark_annual_return = (annual_cumulative - 1).to_dict()  # 转为收益率并存为字典
    
    return benchmark_portfolio_daily, benchmark_annual_return


def ibd_indicator_cal(start_date, end_date, combined_df, benchmark_detail_data, end_date_list, cash=100000000.0):
    """
    增量回测指标计算
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        combined_df: 策略每日价值数据
        benchmark_detail_data: 基准数据
        end_date_list: 结束日期列表
        cash: 初始资金
        
    返回:
        处理后的DataFrame, 指标字典
    """
    # 每日的资产市值的日收益率
    combined_df['value_ratio'] = combined_df['value'] / cash - 1
    
    # 基准累计的收益差
    benchmark_portfolio_daily, benchmark_annual_return = calculate_benchmark_annual_return(
        benchmark_detail_data, start_date, end_date)
    
    bpd = benchmark_portfolio_daily.reset_index()
    combined_df['date'] = bpd['TRADEDATE']
    combined_df['return'] = bpd['return']  # 累计收益率
    combined_df['daily_return'] = bpd['daily_return']  # 每日收益率
    
    # 资产市值与基准累计的收益差
    combined_df['cumdiff'] = combined_df['value_ratio'] - combined_df['return']
    # 每日的资产市值的日收益率
    combined_df['value_daily_ratio'] = combined_df['value'].pct_change()
    # 每日的资产市值与基准的收益差
    combined_df['diff'] = combined_df['value_daily_ratio'] - combined_df['daily_return']
    
    # 提取年份,计算每年年化收益率
    combined_df['date_copy'] = pd.to_datetime(combined_df['date'])
    combined_df['year'] = combined_df['date_copy'].dt.year
    combined_df['annual_multiplier'] = (1 + combined_df['value_daily_ratio']).fillna(1)  # 每日收益转换为乘积形式
    
    # 按年份分组计算每年的累计收益
    annual_cumulative = combined_df.groupby('year')['annual_multiplier'].prod()  # 按年份计算乘积
    annual_returns = (annual_cumulative - 1).to_dict()  # 转为收益率并存为字典
    print(f'策略年华收益率:{annual_returns}')
    
    # 与benchmark比的日胜率、月胜率、年胜率
    combined_df['win_loss_lst'] = [int(e > 0) for e in combined_df['diff']]
    daily_wins = sum(1 for item in combined_df['win_loss_lst'] if item == 1)
    daily_win_ratio_benchmark = daily_wins / len(combined_df['win_loss_lst'])
    
    # 组合相对benchmark月收益差
    monthly_return_diffs = monthly_analysis(combined_df)
    # 组合相对benchmark年收益差
    yearly_return_diffs = yearly_analysis(combined_df)
    
    # 相对benchmark月胜出次数
    monthly_wins = sum(1 for item in monthly_return_diffs if item > 0)
    # 相对benchmark年胜出次数
    yearly_wins = sum(1 for item in yearly_return_diffs if item > 0)
    
    # 月胜率
    monthly_win_ratio_benchmark = monthly_wins / len(monthly_return_diffs) if len(monthly_return_diffs) > 0 else 0
    # 年胜率
    yearly_win_ratio_benchmark = yearly_wins / len(yearly_return_diffs) if len(yearly_return_diffs) > 0 else 0
    
    # 读取获胜次数和毛利润，亏损次数和毛亏损
    combined_dict = {'wins': 0, 'losses': 0, 'gross_profits': 0, 'gross_losses': 0}
    for prev_date in end_date_list:
        try:
            with open(f'state/win_loss_dict{prev_date}.pkl', 'rb') as f:
                win_loss_dict = pickle.load(f)
                combined_dict['wins'] += win_loss_dict.get('wins', 0)
                combined_dict['losses'] += win_loss_dict.get('losses', 0)
                combined_dict['gross_profits'] += win_loss_dict.get('gross_profits', 0)
                combined_dict['gross_losses'] += win_loss_dict.get('gross_losses', 0)
        except (FileNotFoundError, KeyError):
            print(f"未找到{prev_date}的win_loss_dict文件或读取错误")
    
    # 总的交易次数（买卖配对次数）
    total_trades = combined_dict['wins'] + combined_dict['losses']
    # 交易胜率 = 盈利交易次数 / 总的交易次数
    win_ratio = combined_dict['wins'] / total_trades if total_trades > 0 else 0
    # 平均盈利
    avg_profit = combined_dict['gross_profits'] / combined_dict['wins'] if combined_dict['wins'] > 0 else 0
    # 平均亏损
    avg_loss = combined_dict['gross_losses'] / combined_dict['losses'] if combined_dict['losses'] > 0 else 0
    # 盈亏比 = 盈利平均值 / 亏损平均值
    if avg_loss != 0:
        risk_reward_ratio = avg_profit / abs(avg_loss)
    else:
        risk_reward_ratio = 0
    
    # 计算各种指标
    alpha = er.stats.alpha(combined_df['value_daily_ratio'], combined_df['daily_return'])
    beta = er.stats.beta(combined_df['value_daily_ratio'], combined_df['daily_return'])
    base_annual_return = er.stats.annual_return(combined_df['daily_return'])
    annual_return = er.stats.annual_return(combined_df['value_daily_ratio'])
    cum_returns_final = er.stats.cum_returns_final(combined_df['value_daily_ratio'])
    volatility = er.stats.annual_volatility(combined_df['value_daily_ratio'])
    omega_ratio = er.stats.omega_ratio(combined_df['value_daily_ratio'])
    calmar_ratio = er.stats.calmar_ratio(combined_df['value_daily_ratio'])
    sortino_ratio = er.stats.sortino_ratio(combined_df['value_daily_ratio'])
    sharpe_ratio = er.stats.sharpe_ratio(combined_df['value_daily_ratio'])
    max_drawdown = er.stats.max_drawdown(combined_df['value_daily_ratio'])
    max_diff_drawdown = er.stats.max_drawdown(combined_df['diff'])
    
    # 计算平均超额收益
    mean_excess_return = np.nanmean(combined_df['diff'])
    # 计算超额收益的标准差
    std_excess_return = np.nanstd(combined_df['diff'])
    # 计算信息比率 = 平均超额收益 / 超额收益的标准差
    information_ratio = mean_excess_return / std_excess_return if std_excess_return > 0 else 0
    
    # 存为指标字典
    ibd_metrics = {
        'max_drawdown': round(max_drawdown, 3),  # 组合的最大回撤
        'max_diff_drawdown': round(max_diff_drawdown, 3),  # 组合的最大回撤
        'daily_win_ratio_benchmark': round(daily_win_ratio_benchmark, 3),  # 与 benchmark比较的日胜率
        'monthly_win_ratio_benchmark': round(monthly_win_ratio_benchmark, 3),  # 与 benchmark比较的月胜率
        'yearly_win_ratio_benchmark': round(yearly_win_ratio_benchmark, 3),  # 与 benchmark比较的年胜率
        'win_ratio': round(win_ratio, 3),  # 胜率
        'risk_reward_ratio': round(risk_reward_ratio, 3),  # 盈亏比
        'avg_profit': round(avg_profit, 3),
        'avg_loss': round(avg_loss, 3),
        'information_ratio': round(information_ratio, 3),  # 信息比率
        'alpha': round(alpha, 3),  # 阿尔法
        'beta': round(beta, 3),  # 贝塔
        'base_annual_return': round(base_annual_return, 3),  # 基准收益率
        'annual_return': round(annual_return, 3),  # 年化收益率
        'cum_returns_final': round(cum_returns_final, 3),  # 累计收益率
        'volatility': round(volatility, 3),  # 收益波动率
        'omega_ratio': round(omega_ratio, 3),  # omega比率
        'calmar_ratio': round(calmar_ratio, 3),  # calmar 比率
        'sortino_ratio': round(sortino_ratio, 3),  # sortino比率
        'sharpe_ratio': round(sharpe_ratio, 3),  # sharpe比率
    }
    
    return combined_df, ibd_metrics


def save_dict_to_file(dict_data, filename):
    """
    将字典保存到文件
    
    参数:
        dict_data: 要保存的字典
        filename: 文件名
    """
    with open(filename, 'w') as file:
        for key, value in sorted(dict_data.items()):
            file.write(f'{key}:{value}\n')


def combined_two_lists(list1, list2):
    """
    合并两个列表，去除重复项
    
    参数:
        list1: 第一个列表
        list2: 第二个列表
        
    返回:
        合并后的列表
    """
    seen = set()
    combined_list = []
    for item in list1 + list2:
        if item not in seen:
            combined_list.append(item)
            seen.add(item)
    return combined_list



def cvxopt(cur_date, cur_st_list,codeslist,BaseStockDetailData,current_value,stock_dic,FIFO_positions,max_position,constraint_manager=None,strategy=None):
    """
    凸优化调仓
    # cur_date 当前处理的日期
    # cur_st_list st列表
    """
    #从codeslist中排除ST股票，再加上当前日期基准成分股的股票，最后去重并排序
    allcodes=sorted((set(codeslist)-set(cur_st_list)) | set(BaseStockDetailData[BaseStockDetailData['TRADEDATE']==cur_date]['STOCKCODE']))
    print(f"allcodes为:{allcodes}")

    #构建约束检查上下文
    context={
        'strategy':strategy,
        'current_date':cur_date,
        'st_list':cur_st_list
    }
    #预过滤：只保留在stock_dic中的股票
    valid_stocks=[stock for stock in allcodes if stock in stock_dic]
    #约束过滤
    if constraint_manager and context:
        filtered_stocks=constraint_manager.filter_for_buying(valid_stocks,context)
        strategy.log(f"约束过滤：{len(valid_stocks)} -> {len(filtered_stocks)}只股票")
    else:
        filtered_stocks=valid_stocks


    # 当前资产市值  current_value
    # score数据
    expected_returns = []
    # 市值数据
    market_caps = []
    # 行业数据
    industry_mapping = []
    # 股票代码
    codes = []
    for stock in filtered_stocks:
        codes.append(stock)
        expected_returns.append(stock_dic[stock]['score'])
        market_caps.append(stock_dic[stock]['marketvalue'])
        industry_mapping.append(stock_dic[stock]['indecode'])

    # 持仓股票权重
    dict_weights_raw = {}
    # 持仓股票权重
    current_weights_raw = []
    # 市值
    current_market_caps = []
    # 行业
    current_industry_mapping = []
    # 持仓股票代码
    selected_indices = []
    for stock,buy_dates in FIFO_positions.items():
        selected_indices.append(stock)
        bs_size = sum(buy_dates.values())
        bs_price = stock_dic[stock]['price']
        current_industry_mapping.append(stock_dic[stock]['indecode'])
        current_market_caps.append(stock_dic[stock]['marketvalue'])
        w = (bs_price * bs_size) / (
                    current_value * max_position)  # 价格*size/(current_value*self.params.max_position)
        w = round(w,8)
        current_weights_raw.append(w)
        dict_weights_raw[stock] = w
    print('selected_indices:', selected_indices)


    # 筛选符合条件的行
    filtered_data = BaseStockDetailData[
        (BaseStockDetailData['TRADEDATE'] == cur_date) &
        (BaseStockDetailData['STOCKCODE'].notnull()) &
        (BaseStockDetailData['STOCKCODE'].str.len() == 9)
        ]
    # 提取需要的列
    market_caps_benchmark = filtered_data['MARKETVALUE'].tolist()  # 市值
    industry_benchmark = filtered_data['INDCODE'].tolist()  # 行业
    # 成分股权重,权重归一化
    benchmark_weights_raw = filtered_data['WEIGHT'].apply(
        lambda x: x / 100.0 if isinstance(x, float) else 0.0
    ).tolist()
    # 成分股股票代码
    benchmark_indices = []
    for cd in filtered_data['STOCKCODE']:
        benchmark_indices.append(cd)  # 直接添加代码
    print(f'benchmark_indices成分股股票代码:', benchmark_indices)
    print(f'benchmark_indices成分股股票权重:', benchmark_weights_raw)
    print(f'market_caps_benchmark市值:', market_caps_benchmark)
    print(f'industry_benchmark行业:', industry_benchmark)

    # 创建一个DataFrame，包含股票代码对应的得分、市值和行业信息
    section_data = pd.DataFrame(
        {'score': expected_returns, 'marketcaps': market_caps, 'industry': industry_mapping},
        index=codes)

    # 创建一个DataFrame来存储持有股票数据集
    # 该DataFrame包括权重、市值和行业标签，索引为选定的股票代码
    hold_data = pd.DataFrame(
        {'weight': current_weights_raw, 'marketcaps': current_market_caps, 'industry': current_industry_mapping},
        index=selected_indices)

    # 创建一个DataFrame来存储基准数据
    # 包含权重、市值和行业信息
    benchmark_data = pd.DataFrame({'weight': benchmark_weights_raw, 'marketcaps': market_caps_benchmark,
                                   'industry': industry_benchmark}, index=benchmark_indices)

    section_data['score'] = section_data['score'].astype('float64')
    section_data['marketcaps'] = section_data['marketcaps'].astype('float64')
    section_data['industry'] = section_data['industry'].astype('int32')

    hold_data['weight'] = hold_data['weight'].astype('float64')
    hold_data['marketcaps'] = hold_data['marketcaps'].astype('float64')
    hold_data['industry'] = hold_data['industry'].astype('int32')

    benchmark_data['weight'] = benchmark_data['weight'].astype('float64')
    benchmark_data['marketcaps'] = benchmark_data['marketcaps'].astype('float64')
    benchmark_data['industry'] = benchmark_data['industry'].astype('int32')
    # 调用外部优化器计算权重
    optimizer = CvxPortfolioOptimizer(
        weights_upper=1,
        weights_offset=0.01,
        marketcaps_offset=0.1,
        industry_offset=0.01,
        turnover_upper=1
    )

    # 凸优化计算返回候选股票的权重
    result_w = optimizer.optimize(section_data, hold_data, benchmark_data)
    result_w['weight'] = result_w['weight'].round(8)
    return result_w, dict_weights_raw
