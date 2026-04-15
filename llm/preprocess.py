import json
import pathlib
import pandas as pd
import numpy as np
from datetime import datetime
import qlib
from qlib.data import D
data_path="/home/quant/zc/finance_deal/qlib_data/price_data0821"
qlib.init(provider_uri=data_path)

input_root = pathlib.Path(r"/home/quant/zc/backtrader/QuantBacktester_57/llm/input")

class_map = {
    1: "上升区间",
    2: "下降区间",
    3: "峰点",
    4: "谷点",
}
target_cols = ["上升区间", "下降区间", "峰点", "谷点"]

# stock_map = {
#     "五粮液": "000858.SZ",
#     "贵州茅台": "600519.SH",
#     "今世缘": "603369.SH",
#     "泸州老窖": "000568.SZ",
#     "古井贡酒": "000596.SZ",
#     "口子窖": "603589.SH",
#     "老白干酒": "600559.SH",
#     "山西汾酒": "600809.SH",
#     "水井坊": "600779.SH",
#     "迎驾贡酒": "603198.SH",
#     "洋河股份": "002304.SZ",
#     "伊力特": "600197.SH",
# }
# ["600519.SH","000858.SZ","603369.SH","000568.SZ","000596.SZ","603589.SH","600559.SH","600809.SH","600779.SH","603198.SH","002304.SZ","600197.SH"]

stock_map={
    "比亚迪":"002594.SZ",
    "宁德时代": "300750.SZ",
    "欣旺达": "300207.SZ",
    "亿纬锂能": "300014.SZ",
    "珠海冠宇": "688772.SH",
    "孚能科技": "688567.SH",
}
# ["002594.SZ","300750.SZ","300207.SZ","300014.SZ","688772.SH","688567.SH"]

prices = D.features(
                instruments=["002594.SZ","300750.SZ","300207.SZ","300014.SZ","688772.SH","688567.SH"],
                fields=['$close'],
                start_time='2018-01-30',
                end_time='2026-02-01',
                freq='day',
            ).reset_index()
prices.rename(columns={'$close': 'close'}, inplace=True)


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def daily_to_monthly_close(df: pd.DataFrame, date_col: str = 'datetime', close_col: str = 'close') -> pd.Series:
    """
    将日线DataFrame（含datetime列和close列）转换为月线收盘价序列（取月末最后一个交易日价格）。

    参数:
    df (pd.DataFrame): 输入的日线数据，需包含日期列和收盘价列。
    date_col (str): 日期列的列名，默认为'datetime'。
    close_col (str): 收盘价列的列名，默认为'close'。

    返回:
    pd.Series: 月线收盘价序列，索引为月末日期，值为对应月末最后一个交易日的收盘价。
    """
    # 检查输入是否为DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pd.DataFrame类型")
    # 检查必要列是否存在
    missing_cols = [col for col in [date_col, close_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列: {missing_cols}")
    # 复制数据避免修改原DataFrame
    df_copy = df[[date_col, close_col]].copy()
    # 将日期列转换为datetime格式
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    # 删除日期转换失败的行（NaT）
    df_valid = df_copy.dropna(subset=[date_col])
    if df_valid.empty:
        raise ValueError("转换后无有效日期数据")
    # 设置日期为索引
    df_valid.set_index(date_col, inplace=True)
    # 按月重采样，取最后一个交易日的收盘价
    monthly_close = df_valid[close_col].resample('M').last().dropna()
    monthly_close.index=monthly_close.index.strftime("%Y-%m-%d")
    return monthly_close


def calculate_winrate_odds(condition_list, stock_close,trade_direction, h_n=1, max_back=1):
    """
    计算交易策略的胜率、盈亏比和得分。

    参数:
    condition_list (list): 触发交易的索引条件列表。
    stock_close (list/np.array): 股票收盘价序列。
    h_n (int): 持有周期（卖出价格的偏移量）。
    max_back (float): 用于计算得分的分母参数，默认为1。

    返回:
    list: 包含 [h_n, 交易次数, 总次数, 盈利均值, 亏损均值, 胜率, 盈亏比, 得分] 的列表。
    """

    W = 0  # 盈利次数
    r_w = []  # 盈利收益率列表
    r_l = []  # 亏损收益率列表
    valid_count=0
    total_len=len(stock_close)
    # 遍历每一个满足条件的交易点
    for c_idx, cond_idx in enumerate(condition_list):
        # try:
        p_buy = stock_close[cond_idx]
        # 边界检查：确保持有周期后仍有数据
        current_idx=stock_close.index.get_loc(cond_idx)

        if current_idx + h_n >= total_len:
            continue

        p_sell = stock_close.iloc[current_idx+h_n]
        r = (p_sell - p_buy)*trade_direction
        valid_count+=1# 计算价格差
        # 判断盈亏并记录
        if r > 0:
            W += 1
            r_w.append(r / p_buy)
        else:
            r_l.append(r / p_buy)
        # except Exception as e:
        #     print("date error")
    # --- 统计计算 ---
    # 1. 计算胜率
    win_rate = round(W / valid_count, 3) if valid_count > 0 else 0
    # 2. 计算平均盈利收益率
    rw_mean = round(np.mean(r_w), 3) if len(r_w) > 0 else 0
    # 3. 计算平均亏损收益率 (取绝对值)
    rl_mean = round(np.mean(np.abs(r_l)), 3) if len(r_l) > 0 else 0
    # 4. 计算盈亏比 (Odds)
    # 避免除以零的情况
    if len(r_w) == 0 or rw_mean == 0:
        odds = 1e-5
    elif len(r_l) == 0 or rl_mean == 0:
        odds = 9999
    else:
        odds = round(np.mean(r_w) / np.mean(np.abs(r_l)), 3)
    # 5. 计算最终得分
    # 原逻辑似乎是：((胜率 * 平均盈利) - ((1-胜率) * 平均亏损)) / max_back * 100
    scores = round(((win_rate * rw_mean) - ((1 - win_rate) * rl_mean)) / max_back * 100, 3)
    # 返回结果列表
    # (注：原代码中 h_n, W, len(condition_list) 可能是用于后续调试或归档)
    return [h_n, W, valid_count, rw_mean, rl_mean, win_rate, odds, scores]



def clue_weight(winrate_odds):
    if not isinstance(winrate_odds, list) or len(winrate_odds) <= 1:
        return 1.0
    # for row in winrate_odds[1:]:
    #     if not isinstance(row, list) or len(row) < 7:
    #         continue
    #     hold_period = to_float(row[0])
    #     if hold_period != 1:
    #         continue
    row=winrate_odds
    win_count = to_float(row[1])
    total_count = to_float(row[2])
    avg_profit = to_float(row[3])
    avg_loss = to_float(row[4])
    win_rate = to_float(row[5])
    odds = to_float(row[6])
    profit_factor = avg_profit / max(avg_loss, 1e-6)
    confidence = min(1.0, total_count / 15.0)
    edge = win_rate * odds - (1.0 - win_rate)
    quality = edge * (0.6 + 0.4 * confidence) * (0.7 + 0.3 * min(profit_factor, 3.0) / 3.0)
    if win_count is not None and win_count >= 1:
        quality = quality * (0.9 + 0.1 * min(win_count, 10.0) / 10.0)
    # score = 1.0 + quality
    score=quality
    return max(0.1, min(score, 4.0))
    # return 1.0


def clue_weight1(winrate_odds):
    if not isinstance(winrate_odds, list) or len(winrate_odds) <= 7:
        return 1.0

    row = winrate_odds
    total_count = to_float(row[2])  # 总交易次数
    score = to_float(row[7])  # 获取计算出的期望得分 (score)

    if score is None or total_count is None:
        return 1.0

    # 1. 如果期望为负，说明这个线索是亏损的，直接抛弃（权重给0）
    if score <= 0:
        return 0.0

    # 2. 置信度惩罚：交易次数越少，越容易是运气，所以打个折扣 (比如满分需要 15 次以上)
    confidence = min(1.0, total_count / 15.0)

    # 3. 基础权重基于得分，并结合置信度
    # score 已经是百分比形式 (乘了100)，比如 0.5% 就是 0.5
    # 我们将 score 放缩到合适的权重区间，比如 1个百分点的期望算 1.0 的质量
    quality = score * confidence

    # 4. 计算最终权重 (给一个基础权重 0.5，加上 quality，并限制最大不超过 3.0)
    final_weight = 0.5 + quality

    return max(0.0, min(final_weight, 3.0))

def build_one(json_path: pathlib.Path):
    out_path = json_path.parent / "back_tests_winloss.csv"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    name=json_path.parent.name
    stockcode=stock_map.get(name)
    print(stockcode)
    price=prices[prices['instrument']==stockcode]
    rows = []
    for item in data:
        cls = item.get("class_")
        signal = class_map.get(cls)
        if signal is None:
            continue
        trade_direction=1 if cls in (1,4) else -1
        stock_close=daily_to_monthly_close(price)
        dates_list=item.get("special_dates")
        cutoff_date = datetime.strptime("2024-10-30", "%Y-%m-%d")
        limit_date=datetime.strptime("2015-10-30", "%Y-%m-%d")
        filtered_dates = [
            d for d in dates_list
            if datetime.strptime(d, "%Y-%m-%d") < cutoff_date
        ]
        winrate_odds=calculate_winrate_odds(filtered_dates,stock_close,trade_direction)
        # weight = clue_weight(item.get("winrate_odds"))
        weight = clue_weight1(winrate_odds)
        actual_dates= [
            d for d in dates_list
            # if datetime.strptime(d, "%Y-%m-%d") >= cutoff_date
        ]
        for d in actual_dates:
            rows.append({"日期": pd.to_datetime(d), "信号": signal, "权重": weight})

    df = pd.DataFrame(rows)
    weighted = (
        df.pivot_table(index="日期", columns="信号", values="权重", aggfunc="sum", fill_value=0.0)
        if not df.empty
        else pd.DataFrame(columns=target_cols)
    )
    # if not df.empty:
    #     sum_tbl = df.pivot_table(index="日期", columns="信号", values="权重", aggfunc="sum", fill_value=0.0)
    #     cnt_tbl = df.pivot_table(index="日期", columns="信号", values="权重", aggfunc="count", fill_value=0.0)
    #     weighted = sum_tbl / np.sqrt(cnt_tbl.where(cnt_tbl > 0, 1.0))
    # else:
    #     weighted = pd.DataFrame(columns=target_cols)


    for c in target_cols:
        if c not in weighted.columns:
            weighted[c] = 0.0
    weighted = weighted[target_cols]

    if not df.empty:
        full_month_ends = pd.date_range(
            df["日期"].min().to_period("M").to_timestamp("M"),
            df["日期"].max().to_period("M").to_timestamp("M"),
            freq="M",
        )
        out = weighted.reindex(full_month_ends, fill_value=0.0).reset_index().rename(columns={"index": "日期"})
        out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    else:
        out = pd.DataFrame(columns=["日期"] + target_cols)

    for c in target_cols:
        out[c] = out[c].astype(float).round(4)

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, len(out)


json_files = sorted(input_root.glob("*/strategy_clues.json"))
if not json_files and (input_root / "strategy_clues.json").exists():
    json_files = [input_root / "strategy_clues.json"]

for p in json_files:
    s1=stock_map.keys()
    if p.parent.name in stock_map.keys():
        out_path, row_count = build_one(p)
        print(f"已生成: {out_path}, 行数: {row_count}")






