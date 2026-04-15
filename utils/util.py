import pandas as pd
import json
import os
import sys
from collections import defaultdict

def process_section_codes(folder_path, section_codes_path, output_path):
    """
    处理股票池代码映射的函数

    Args:
        folder_path: 包含CSV文件的文件夹路径
        section_codes_path: 板块代码文件的路径
        output_path: 输出文件的路径

    Returns:
        pandas.Series: 处理后的股票代码序列
    """
    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 创建代码映射字典
    code_mapping = {}
    for code in csv_files:
        # 跳过长度超过14或首字母为'T'的文件
        if len(code) > 14 or code[:1] == 'T':
            continue

        first_six = int(code[:6])
        first_nine = code[:9]
        code_mapping[first_six] = first_nine

    # 读取板块代码文件
    section_code = pd.read_csv(section_codes_path)
    # 应用映射
    section_code['stockcode'] = section_code['000001'].map(code_mapping)

    # 检查是否有NaN值
    nan_count = section_code['stockcode'].isna().sum()
    if nan_count > 0:
        print(f"警告: 有 {nan_count} 个代码未能成功映射")

        # 找出未能映射的代码
        failed_codes = section_code[section_code['stockcode'].isna()]['000001'].tolist()
        print(f"未能映射的代码: {failed_codes}")
    #删除包含NaN的行
    section_code = section_code.dropna(subset=['stockcode'])

    # 保存结果
    section_code['stockcode'].to_csv(output_path, index=False)

    print(f"处理完成！结果已保存到: {output_path}")

    return section_code['stockcode']


def simple_compare(portfolio1, portfolio2):
    """简化版比对，只检查关键字段"""

    # 创建简化版的持仓（移除price和dividend_cash）
    def simplify_portfolio(portfolio):
        simplified = {}
        for stock_code, positions in portfolio.items():
            simplified[stock_code] = {}
            for buy_date, position in positions.items():
                simplified[stock_code][buy_date] = {
                    'stock_code': position['stock_code'],
                    'buy_date': position['buy_date'],
                    'size': position['size']
                }
        return simplified

    simplified1 = simplify_portfolio(portfolio1)
    simplified2 = simplify_portfolio(portfolio2)

    return simplified1 == simplified2

def adjust_records_conversion():
    #从调仓表出发得到持仓字典
    # 读取调整记录
    adj_record = pd.read_csv("result/adjust_records_copy.csv")
    cur_date='2025-09-10'
    # 读取实际持仓状态
    with open(f"state/portfolio{cur_date}.json", 'r') as f:
        portfolio_state = json.load(f)

    # 处理日期格式
    adj_record['date'] = pd.to_datetime(adj_record['date'], format='%Y-%m-%d')
    target_date = pd.to_datetime(cur_date, format='%Y-%m-%d')

    # 筛选目标日期前的记录
    records_before_target = adj_record[adj_record['date'] <= target_date].copy()

    # 初始化持仓字典
    portfolio = defaultdict(dict)

    # 处理每一条交易记录
    for _, row in records_before_target.iterrows():
        stock_code = row['stock']
        operation = row['operation']
        amount = row['amount']
        trade_date = row['date'].strftime('%Y-%m-%d')

        if operation == 'buy':
            if trade_date not in portfolio[stock_code]:
                portfolio[stock_code][trade_date] = {
                    'stock_code': stock_code,
                    'buy_date': trade_date,
                    'size': 0,
                    'price': 0,
                    'dividend_cash': 0.0
                }

            # 累计买入数量
            portfolio[stock_code][trade_date]['size'] += amount

        elif operation == 'sell':
            remaining_sell = abs(amount)

            # 获取该股票所有买入记录，按日期排序（先进先出）
            buy_dates = sorted(portfolio[stock_code].keys())

            for buy_date in buy_dates:
                if remaining_sell <= 0:
                    break

                position = portfolio[stock_code][buy_date]
                if position['size'] > 0:
                    position['size'] -= remaining_sell
                    remaining_sell = 0
                    if position['size'] == 0:
                        del portfolio[stock_code][buy_date]

    # 清理空持仓
    stocks_to_remove = []
    for stock_code, positions in portfolio.items():
        dates_to_remove = []
        for buy_date, position in positions.items():
            if position['size'] <= 0:
                dates_to_remove.append(buy_date)
        for date in dates_to_remove:
            del positions[date]

        if not positions:
            stocks_to_remove.append(stock_code)

    for stock_code in stocks_to_remove:
        del portfolio[stock_code]

    portfolio_dict = {k: dict(v) for k, v in portfolio.items()}


    # 使用简化版比对
    if simple_compare(portfolio_dict, portfolio_state):
        print("✅ 两个持仓完全一致（忽略price和dividend_cash）")
        with open(f"state/portfolio{cur_date}.json", 'w') as f:
            json.dump(portfolio_state, f)
            print(f"调仓表转换Portfolio状态已保存")
    else:
        print("❌ 两个持仓存在差异")


def portfolio_df_to_dict(portfolio_df):
    """
    将持仓DataFrame转换为字典格式

    Args:
        portfolio_df: 包含持仓信息的DataFrame

    Returns:
        dict: 转换后的持仓字典
    """
    # 确保日期格式正确
    if 'buy_date' in portfolio_df.columns:
        portfolio_df['buy_date'] = pd.to_datetime(portfolio_df['buy_date']).dt.strftime('%Y-%m-%d')

    # 初始化结果字典
    portfolio_dict = {}

    # 按股票代码分组处理
    for stock_code, group in portfolio_df.groupby('stock_code'):
        portfolio_dict[stock_code] = {}

        # 处理该股票的每一行持仓记录
        for _, row in group.iterrows():
            buy_date = row['buy_date']

            portfolio_dict[stock_code][buy_date] = {
                'stock_code': stock_code,
                'buy_date': buy_date,
                'size': row['size'],
                'price': row.get('price', 0),  # 使用get方法避免KeyError
                'dividend_cash': row.get('dividend_cash', 0.0)
            }

    return portfolio_dict


def portfolio_dict_to_df(portfolio_dict):
    """
    将持仓字典转换为DataFrame

    Args:
        portfolio_dict: 持仓字典

    Returns:
        pandas.DataFrame: 转换后的持仓表
    """
    records = []

    for stock_code, positions in portfolio_dict.items():
        for buy_date, position in positions.items():
            records.append({
                'stock_code': stock_code,
                'buy_date': buy_date,
                'size': position['size'],
                'price': position.get('price', 0),
                'dividend_cash': position.get('dividend_cash', 0.0)
            })

    # 创建DataFrame
    portfolio_df = pd.DataFrame(records)

    # 按股票代码和买入日期排序
    portfolio_df = portfolio_df.sort_values(['stock_code', 'buy_date'])

    return portfolio_df


def cash_value_save_state(cash,value):
    # 保存策略状态
    state = {
        'cash': cash,
        'value': value,
    }

    return state



if __name__ == "__main__":
    # adjust_records_conversion()
    # main()

    # folder_path = '/home/quant/data_test/csv_data/daily_data'
    # section_codes_path = "/home/quant/zc/QuantBacktester_60/input/section_codes.csv"
    # output_path = "/home/quant/zc/QuantBacktester_60/input/section_codes1.csv"
    #
    # result = process_section_codes(folder_path, section_codes_path, output_path)
    # print(f"处理后的股票代码数量: {len(result)}")

    with open(f"../state/portfolio2025-07-18.json", 'r') as f:
        portfolio_state = json.load(f)
    df=portfolio_dict_to_df(portfolio_state)
    df.to_csv(f"../state/portfolio2025-07-18.csv",index=False)
    # compare_dict=portfolio_df_to_dict(df)
    # if simple_compare(portfolio_state, compare_dict):
    #     print("✅ 两个持仓完全一致（忽略price和dividend_cash）")
    # else:
    #     print("❌ 两个持仓存在差异")
