import pandas as pd
import os
import logging
import subprocess
from typing import Dict, List, Tuple,Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def process_trade_log(input_file: str = None,
                      input_df : Optional[pd.DataFrame]=None,
                      output_file: str = None) -> None:
    """
    处理交易日志，计算每日每支股票的权重并输出到CSV文件

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    try:
        if input_file is not None:
            logging.info(f"开始处理交易日志: {input_file}")

            # 读取和预处理数据
            df = pd.read_csv(input_file, header=None,
                             names=['date', 'operation', 'stock', 'price', 'amount', 'factor'])
        else:
            logging.info('开始处理直接输入的DataFrame数据')
            df=input_df.copy()

        # 日期处理 - 使用pandas的向量化操作
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')

        # 股票代码处理
        df['stock'] = df['stock'].str.strip()
        df = df[df['stock'].str.endswith(('.SH', '.SZ'))]

        # 计算交易金额
        df['money'] = df['price'] * df['amount']

        # 使用向量化操作计算买入卖出
        # 创建买入和卖出的掩码
        buy_mask = df['operation'] == 'buy'

        # 使用groupby和agg进行计算
        net_values = df.groupby(['date', 'stock']).apply(
            lambda x: x.loc[buy_mask, 'money'].sum() - x.loc[~buy_mask, 'money'].sum()
        ).reset_index(name='net_value')

        # 只保留为正的记录
        net_values = net_values[net_values['net_value'] > 0]

        # 计算每日权重
        net_values['total_daily_value'] = net_values.groupby('date')['net_value'].transform('sum')
        net_values['weight'] = net_values['net_value'] / net_values['total_daily_value']

        # 创建结果DataFrame
        result_df = net_values[['date', 'stock', 'weight']]

        # 创建一个包含所有日期和股票组合的DataFrame
        all_combinations = df[['date', 'stock']].drop_duplicates()

        # 合并并填充缺失值
        final_result = pd.merge(all_combinations, result_df, on=['date', 'stock'], how='left')
        final_result['weight'] = final_result['weight'].fillna(0)

        # 输出结果
        final_result.to_csv(output_file, index=False)
        logging.info(f"交易日志处理完成，结果已保存至: {output_file}")

    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
    except pd.errors.EmptyDataError:
        logging.error(f"输入文件为空: {input_file}")
    except Exception as e:
        logging.error(f"处理交易日志时发生错误: {e}")

def run():
    #转换数据得到调仓表文件
    #提供input-file
    process_trade_log(input_file="input/trade_log_25071501.txt", output_file="/home/quant/zc/backtrader/QuantBacktester/input/trade_log.csv")
    #提供dataframe
    # process_trade_log(input_df=pd.DataFrame(),output_file="/home/quant/zc/backtrader/QuantBacktester/input/trade_log.csv")

    # 构建并执行命令
    cmd = [
        'python', '/home/quant/zc/backtrader/QuantBacktester/main.py',
        '--mode', 'rebalanceTable',
        '--rebalance_file', './input/trade_log.csv',
    ]

    print(f"\n执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("命令执行成功!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误码: {e.returncode}")
        print(f"错误输出:\n{e.stderr}")
    except Exception as e:
        print(f"执行过程中发生未知错误: {str(e)}")


if __name__ == "__main__":
    run()
