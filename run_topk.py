
import pandas as pd
import os
import logging
import json
import pickle
import sys
import argparse
import time
sys.path.append("/home/quant/zc/QuantBacktester_60")
from pathlib import Path
from main import BacktestManager
from utils.config import config
from utils.util import portfolio_df_to_dict,cash_value_save_state


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(is_ts):
    """
    Loads the dataset from the pickle file.
    """
    if is_ts:
        with open("/home/quant/qlib/dataset_ts.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset
    else:
        with open("dataset.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset


def load_model(model_name):
    """
    Loads the model from the pickle file.
    """
    model_name = "/home/quant/qlib/saved_models/" + model_name + ".pkl"
    with open(model_name, "rb") as file_model:
        model = pickle.load(file_model)
    return model


def load_data_all(is_ts):
    """
    Loads the dataset from the pickle file.
    """
    if is_ts:
        with open("model_input/" + "dataset_ts_all.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset
    else:
        with open("dataset_all.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)
        return dataset


def get_pred_scores(modelname, is_ts):
    """
    Gets the predictions and scores from the model.
    """
    model = load_model(modelname)
    if "_all" in modelname:
        dataset = load_data_all(is_ts)
    else:
        dataset = load_data(is_ts)
    pred_score = model.predict(dataset)
    pred_score = pred_score.rename('score')
    pred_score = pred_score.to_frame('score')
    pred_score = pred_score.reset_index()
    return pred_score


def run():
    # 主程序入口
    pred_score = get_pred_scores("gru_company", True)

    # 可选：保存结果到文件
    pred_score.to_csv("/home/quant/zc/QuantBacktester_60/input/prediction_results.csv", index=False) #这个路径不变
    print("预测结果已保存到 prediction_results.csv")

    """主程序入口"""
    parser = argparse.ArgumentParser(description='量化交易回测系统')

    # 添加参数
    parser.add_argument('--mode', type=str, default='topk',help='回测模式')
    parser.add_argument('--predict', type=str, default='/home/quant/zc/QuantBacktester_60/input/prediction_results.csv', help='预测数据文件路径')
    parser.add_argument('--pool', type=str, default='/home/quant/zc/QuantBacktester_60/input/section_codes1.csv', help='股票池文件路径')
    parser.add_argument('--output', type=str, default='result/output.txt', help='输出结果文件路径')  # output.txt
    parser.add_argument('--topk_adjust', type=str, default='result/adjust_records.csv', help='topk调仓结果文件路径')
    parser.add_argument('--plot_output', type=str, default='plot/strategy_plot1.png', help='策略回测图片保存路径')
    parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')

    # 解析参数
    args = parser.parse_args()
    # 初始化回测管理器
    backtest_manager = BacktestManager()

    cur_date = '2025-07-18'  #持仓表对应的日期
    cash=6987991.265966129
    value=121545694.26596613
    df=pd.read_csv("state/portfolio2025-07-18.csv")   #当天买卖完成后的持仓表
    portfolio_dict = portfolio_df_to_dict(df)
    state = cash_value_save_state(cash=cash, value=value)
    # 写入文件
    with open(f'state/state{cur_date}.pkl', 'wb') as f:
        pickle.dump(state, f)
    with open(f"state/portfolio{cur_date}.json", 'w') as f:
        json.dump(portfolio_dict, f)
    print(f"现金文件和持仓表转换存储完成")

    # 更新策略配置参数
    param_map={
        'start_date': cur_date,  #回测从2025-07-18到2025-09-10结束
        'end_date': '2025-09-10',
        'deal_dividend':  False,   #分红
        'save_state':  True,   #预测:True，回测：False
        'adjust_freq': 'D',  #调仓频率：'D'=日频，'M'=月频，'W'=周频，'Q'=季频
        'initial_stocks': 50,  #初始建仓股票数量
        'adjust_count': 5,   # 每次调仓的股票数量
        'min_holding_days': 30,  # 最小持有天数,设置为0表示关闭持有时间约束
        'stock_pool_mode': 1,  # 股票池模式：0=关闭约束，1=启用约束
        'market_cap_threshold':50 , # 市值检查阈值(亿元),设置为0表示关闭市值检查
        'consecutive_limit_days': 2, # 连续涨跌停天数限制,设置为0表示关闭约束
        'st_stock_mode': 1, # ST股票模式：0=关闭约束，1=启用约束
        'enable_rebalance': False,  # 是否启用再平衡功能，True启用，False禁用
        'is_predict': True,   #预测:True，回测：False
    }

    strategy_params={k:v for k,v in param_map.items() if v is not None}
    if strategy_params:
        print("\n=== 策略配置参数更新 ===")
        config.update_strategy_params(**strategy_params)
        print("=======================")


    # 设置输出文件
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='utf-8')

    # 记录开始时间
    start_time = time.time()

    # 执行回测
    print("执行TOPK回测...")
    backtest_manager.run_topk_backtest(args.predict, args.pool, args.topk_adjust, args.plot_output, args.verbose)

    # 输出总运行时间
    end_time = time.time()
    print(f'回测总运行时间：{end_time - start_time:.2f}秒')

    # 关闭输出文件
    if args.output:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    run()
