from pathlib import Path
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss,recall_score

import qlib
from qlib.data import D
provider_uri = os.environ.get('QLIB_PROVIDER_URI', '/home/quant/zc/finance_deal/qlib_data/price_data0821')
qlib.init(provider_uri=provider_uri)


"""
单股处理：
空仓且买信号更强，action=1，
持仓且卖信号更强，action=-1，
score=buy_signal-sell_signal

组合构建：
max_positions,最大持仓数量3
日期内按score从高到低排序
先卖后买---卖：action=-1且在持仓中
         买：持仓股票少于3，action=1的股票中按score买入
权重：1/max_positions
"""


FEATURE_COLS = ["上升区间", "下降区间", "峰点", "谷点"]
BUY_W = np.array([0.4, 0, 0, 0.6])
SELL_W = np.array([0, 0.4, 0.6, 0])
SCORE_W=np.array([0.4,-0.4,-0.6,0.6])


def build_forward_return_samples(feature_table: pd.DataFrame, open_table: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_table.copy()
    feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])
    rebalance_dates = sorted(feature_df["datetime"].dropna().unique().tolist())
    if len(rebalance_dates) < 2:
        return pd.DataFrame(columns=["datetime", "instrument"] + FEATURE_COLS + ["future_return"])
    trade_dates = open_table.index.sort_values()

    def next_trade_date(dt: pd.Timestamp):
        pos = trade_dates.searchsorted(dt, side="right")
        if pos >= len(trade_dates):
            return None
        return trade_dates[pos]

    records = []
    for i in range(len(rebalance_dates) - 1):
        dt = pd.to_datetime(rebalance_dates[i])
        next_dt = pd.to_datetime(rebalance_dates[i + 1])
        entry_dt = next_trade_date(dt)
        exit_dt = next_trade_date(next_dt)
        if entry_dt is None or exit_dt is None:
            continue
        day_df = feature_df[feature_df["datetime"] == dt]
        start_open = open_table.loc[entry_dt]
        end_open = open_table.loc[exit_dt]
        for _, row in day_df.iterrows():
            p0 = start_open.get(row["instrument"], np.nan)
            p1 = end_open.get(row["instrument"], np.nan)
            if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                records.append(
                    {
                        "datetime": dt,
                        "instrument": row["instrument"],
                        FEATURE_COLS[0]: row[FEATURE_COLS[0]],
                        FEATURE_COLS[1]: row[FEATURE_COLS[1]],
                        FEATURE_COLS[2]: row[FEATURE_COLS[2]],
                        FEATURE_COLS[3]: row[FEATURE_COLS[3]],
                        "future_return": float(p1 / p0 - 1.0),
                    }
                )
    return pd.DataFrame(records)


def build_single_feature_table(
    csv_path: Path, instrument_name: str, calendar_table: pd.DataFrame
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "未知区间" in df.columns:
        df.drop("未知区间", axis=1, inplace=True)
    df = df.rename(columns={"日期": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = rolling_zscore_normalize(df, FEATURE_COLS)
    calendar_df = calendar_table.copy()
    calendar_df["datetime"] = pd.to_datetime(calendar_df["datetime"])
    calendar_df["month"] = calendar_df["datetime"].dt.to_period("M")
    month_last_trading_day = (
        calendar_df.groupby("month", as_index=False)["datetime"].max().rename(
            columns={"datetime": "mapped_datetime"}
        )
    )
    df["month"] = df["datetime"].dt.to_period("M")
    month_features = pd.merge(
        df[["month"] + FEATURE_COLS],
        month_last_trading_day,
        on="month",
        how="inner",
    )
    month_features = (
        month_features.sort_values(["mapped_datetime"])
        .drop_duplicates(subset=["mapped_datetime"], keep="last")
        .rename(columns={"mapped_datetime": "datetime"})
    )
    month_features["instrument"] = instrument_name
    return month_features[["datetime", "instrument"] + FEATURE_COLS]


def fit_score_weights_linear_regression(
    samples_df: pd.DataFrame,
    train_end_date: pd.Timestamp,
    max_iter: int = 3000,
    l2_reg: float = 1e-4,
) -> dict:
    df = samples_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["profit_label"] = (df["future_return"] > 0).astype(float)
    train_df = df[df["datetime"] < train_end_date]
    test_df = df[df["datetime"] > train_end_date]
    if train_df.empty:
        raise ValueError("训练集为空，无法拟合score权重")
    x_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["profit_label"].to_numpy(dtype=float)
    c_value = 1.0 / l2_reg if l2_reg > 0 else 1e12
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced"
    )
    model.fit(x_train, y_train)
    weights = model.coef_.reshape(-1)

    train_prob = model.predict_proba(x_train)[:, 1]
    train_pred = (train_prob >= 0.5).astype(float)
    train_acc = float(accuracy_score(y_train, train_pred))
    train_logloss = float(log_loss(y_train, train_prob, labels=[0.0, 1.0]))
    test_acc = float("nan")
    test_logloss = float("nan")
    if not test_df.empty:
        x_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
        y_test = test_df["profit_label"].to_numpy(dtype=float)
        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(float)
        test_acc = float(accuracy_score(y_test, test_pred))
        test_recall=float(recall_score(y_test,test_pred))
        test_logloss = float(log_loss(y_test, test_prob, labels=[0.0, 1.0]))
    print(f"{df['instrument'][0]},test_acc:{test_acc:.2f},test_recall:{test_recall:.2f}")
    pred=np.hstack((train_pred,test_pred))
    return {
        "weights": weights,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_r2": train_acc,
        "test_r2": test_acc,
        "test_pred":pred
    }


def min_max_normalize(df: pd.DataFrame, columns: list,window_months:int=12,neutral_value:float=0.5) -> pd.DataFrame:
    normalized = df.copy()
    for col in columns:
        col_values = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0).astype(float)
        running_min=col_values.rolling(window=window_months,min_periods=1).min()
        running_max = col_values.rolling(window=window_months,min_periods=1).max()
        denominator=running_max-running_min
        normalized[col]=np.where(denominator==0,neutral_value,(col_values-running_min)/denominator,)
    return normalized


def rolling_zscore_normalize(
    df: pd.DataFrame,
    columns: list,
    window_months: int = 12,
    neutral_value: float = 0.0,
    clip_std: float = 3.0,
    preserve_non_negative:bool=True
) -> pd.DataFrame:
    normalized = df.copy()
    for col in columns:
        col_values = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0).astype(float)
        rolling_mean = col_values.rolling(window=window_months, min_periods=1).mean()
        rolling_std = col_values.rolling(window=window_months, min_periods=1).std(ddof=0)
        # rolling_mean=col_values.mean()
        # rolling_std=col_values.std(ddof=0)
        normalized_col = np.where(
            rolling_std == 0.0,
            neutral_value,
            (col_values - rolling_mean) / rolling_std,
        )
        normalized_series=pd.Series(normalized_col, index=normalized.index).clip(-clip_std, clip_std)
        if preserve_non_negative:
            # positive_mask=col_values>0
            # scaled_series=(normalized_series+clip_std)/(2*clip_std)
            # scaled_series=scaled_series.clip(lower=0,upper=1)
            # normalized_series=pd.Series(np.where(positive_mask,scaled_series,0),index=normalized.index)
            # normalized_series=pd.Series(np.where(positive_mask&(rolling_std==0),1,normalized_series),index=normalized.index)
            # normalized_series=normalized_series.clip(lower=0)
            normalized_series=normalized_series.mask(col_values<=0,0)
            normalized_series=normalized_series.clip(lower=0)
        normalized[col]=normalized_series
    return normalized


def add_trading_columns(mapped_signals_df: pd.DataFrame) -> pd.DataFrame:
    df = mapped_signals_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["action"] = 0
    df["weight"] = 0.0
    in_position = False

    for i in range(len(df)):
        # buy_val = df.loc[i, "buy_signal"]
        # sell_val = df.loc[i, "sell_signal"]
        # has_buy = pd.notna(buy_val) and buy_val > 0
        # has_sell = pd.notna(sell_val) and sell_val > 0
        score_val = df.loc[i, "score"]
        has_buy = score_val > 0
        has_sell = score_val <= 0

        if in_position:
            # if has_sell and (not has_buy or sell_val > buy_val):
            if has_sell:
                df.loc[i, "action"] = -1
                df.loc[i, "weight"] = 1.0
                in_position = False
        else:
            # if has_buy and (not has_sell or buy_val > sell_val):
            if has_buy:
                df.loc[i, "action"] = 1
                df.loc[i, "weight"] = 1.0
                in_position = True
    return df


def build_calendar(input_root: Path, calendar_path: Optional[Path]) -> pd.DataFrame:
    if calendar_path is not None and calendar_path.exists():
        calendar_df = pd.read_csv(calendar_path, sep=r"\s+", header=None, names=["datetime"])
        calendar_df["datetime"] = pd.to_datetime(calendar_df["datetime"])
        return calendar_df

    all_monthly_dates = []
    for csv_file in sorted(input_root.glob("*/back_tests.csv")):
        temp_df = pd.read_csv(csv_file, usecols=["日期"])
        all_monthly_dates.append(pd.to_datetime(temp_df["日期"]))

    if not all_monthly_dates:
        raise ValueError(f"未在目录中找到 back_tests.csv: {input_root}")

    all_monthly_dates = pd.concat(all_monthly_dates, ignore_index=True)
    start_dt = all_monthly_dates.min()
    end_dt = all_monthly_dates.max()
    return pd.DataFrame({"datetime": pd.bdate_range(start=start_dt, end=end_dt)})


def build_single_signal_table(
    csv_path: Path, instrument_name: str, calendar_table: pd.DataFrame,score_w=None,test_pred=None
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if '未知区间' in df.columns:
        df.drop('未知区间',axis=1,inplace=True)
    df = df.rename(columns={"日期": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    # df = min_max_normalize(df, FEATURE_COLS)
    df=rolling_zscore_normalize(df,FEATURE_COLS)
    values = df[FEATURE_COLS].values
    df["buy_signal"] = (values * BUY_W).sum(axis=1)
    df["sell_signal"] = (values * SELL_W).sum(axis=1)
    # if test_pred is not None:
    #     length=len(df)-len(test_pred)
    #     df=df.iloc[length:]
    #     df['score']=test_pred
    if score_w is None:
        df['score']=(values*SCORE_W).sum(axis=1)
    else:
        df['score'] = (values * score_w).sum(axis=1)
    calendar_df = calendar_table.copy()
    calendar_df["datetime"] = pd.to_datetime(calendar_df["datetime"])
    calendar_df["month"] = calendar_df["datetime"].dt.to_period("M")

    month_last_trading_day = (
        calendar_df.groupby("month", as_index=False)["datetime"].max().rename(
            columns={"datetime": "mapped_datetime"}
        )
    )

    df["month"] = df["datetime"].dt.to_period("M")
    month_signals = pd.merge(
        df[["month", "buy_signal", "sell_signal","score"]],
        month_last_trading_day,
        on="month",
        how="inner",
    )
    month_signals = (
        month_signals.sort_values(["mapped_datetime"])
        .drop_duplicates(subset=["mapped_datetime"], keep="last")
        .rename(columns={"mapped_datetime": "datetime"})
    )

    #扩展到日频
    # mapped = pd.merge(
    #     calendar_df[["datetime"]],
    #     month_signals[["datetime", "buy_signal", "sell_signal"]],
    #     on="datetime",
    #     how="left",
    # )
    mapped=month_signals
    mapped = mapped[mapped['datetime'] >= "2024-10-30"]
    traded = add_trading_columns(mapped)
    # traded["score"] = traded["buy_signal"].fillna(0) - traded["sell_signal"].fillna(0)
    # traded["score"]=-traded["score"]
    traded["instrument"] = instrument_name
    traded = traded.round(3)
    return traded[["datetime", "instrument", "buy_signal", "sell_signal", "score", "action", "weight"]]


def build_single_signal_table_up_valley(
    csv_path: Path,
    instrument_name: str,
    calendar_table: pd.DataFrame,
    up_weight: float = 0.4,
    valley_weight: float = 0.6,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "未知区间" in df.columns:
        df.drop("未知区间", axis=1, inplace=True)
    df = df.rename(columns={"日期": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    up_col = pd.to_numeric(df.get("上升区间", 0.0), errors="coerce").fillna(0.0).astype(float)
    valley_col = pd.to_numeric(df.get("谷点", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["buy_signal"] = up_weight * up_col + valley_weight * valley_col
    df["score"] = df["buy_signal"]
    calendar_df = calendar_table.copy()
    calendar_df["datetime"] = pd.to_datetime(calendar_df["datetime"])
    calendar_df["month"] = calendar_df["datetime"].dt.to_period("M")
    month_last_trading_day = (
        calendar_df.groupby("month", as_index=False)["datetime"].max().rename(columns={"datetime": "mapped_datetime"})
    )
    df["month"] = df["datetime"].dt.to_period("M")
    month_signals = pd.merge(
        df[["month", "buy_signal", "score"]],
        month_last_trading_day,
        on="month",
        how="inner",
    )
    month_signals = (
        month_signals.sort_values(["mapped_datetime"])
        .drop_duplicates(subset=["mapped_datetime"], keep="last")
        .rename(columns={"mapped_datetime": "datetime"})
    )
    traded = month_signals.copy()
    traded["instrument"] = instrument_name
    traded["sell_signal"] = 0.0
    traded["action"] = 0
    traded["weight"] = 0.0
    traded=traded.round(3)
    return traded[["datetime", "instrument", "buy_signal", "sell_signal", "score", "action", "weight"]]


# def build_portfolio_orders(signal_table: pd.DataFrame, max_positions: int = 3) -> pd.DataFrame:
#     unit_weight = 1.0 / float(max_positions)
#     positions = {}
#     orders = []
#
#     for dt, day_df in signal_table.groupby("datetime", sort=True):
#         day_df = day_df.sort_values("score", ascending=False)
#         sell_rows = day_df[(day_df["action"] == -1) & (day_df["instrument"].isin(positions))]
#         for row in sell_rows.itertuples():
#             exit_weight = positions.pop(row.instrument)
#             orders.append(
#                 {
#                     "datetime": dt,
#                     "instrument": row.instrument,
#                     "action": -1,
#                     "weight": exit_weight,
#                     "score": row.score,
#                 }
#             )
#
#         open_slots = max_positions - len(positions)
#         if open_slots <= 0:
#             continue
#
#         buy_rows = day_df[(day_df["action"] == 1) & (~day_df["instrument"].isin(positions))]
#         for row in buy_rows.head(open_slots).itertuples():
#             positions[row.instrument] = unit_weight
#             orders.append(
#                 {
#                     "datetime": dt,
#                     "instrument": row.instrument,
#                     "action": 1,
#                     "weight": unit_weight,
#                     "score": row.score,
#                 }
#             )
#     return pd.DataFrame(orders)


# def build_portfolio_orders(signal_table: pd.DataFrame, max_positions: int = 3) -> pd.DataFrame:
#     if max_positions<=0:
#         raise ValueError("max_positions必须大于0")
#     unit_weight = 1.0 / float(max_positions)
#     positions = set()
#     orders = []
#     signal_df=signal_table.copy()
#     signal_df['datetime']=pd.to_datetime(signal_df['datetime'])
#     signal_df['month']=signal_df['datetime'].dt.to_period("M")
#     rebalance_dates=(
#         signal_df.groupby("month",as_index=False)['datetime'].max().rename(columns={"datetime":"rebalance_dt"})
#     )
#
#     for dt in rebalance_dates['rebalance_dt']:
#         day_df=signal_df[signal_df["datetime"]==dt]
#         day_df=day_df.sort_values(["score","instrument"],ascending=[False,True])
#         target_positions=set(day_df.head(max_positions)['instrument'].tolist())
#         score_map=day_df.set_index("instrument")['score'].to_dict()
#         if dt.strftime('%Y-%m-%d')=="2024-02-29":
#             print("ok")
#         for instrument in sorted(positions-target_positions):
#             orders.append(
#                 {
#                     "datetime": dt,
#                     "instrument": instrument,
#                     "action": -1,
#                     "weight": unit_weight,
#                     "score": score_map.get(instrument,0),
#                 }
#             )
#         for instrument in sorted(target_positions-positions):
#             orders.append(
#                 {
#                     "datetime": dt,
#                     "instrument": instrument,
#                     "action": 1,
#                     "weight": unit_weight,
#                     "score": score_map.get(instrument, 0),
#                 }
#             )
#         positions=target_positions
#
#     orders_df=pd.DataFrame(orders)
#     # orders_df=orders_df.sort_values(["datetime","action","instrument"],ascending=[True,True,True]).reset_index(drop=True)
#
#     return orders_df


def build_portfolio_orders(signal_table: pd.DataFrame,max_positions=4) -> pd.DataFrame:
    positions={}
    orders = []
    signal_df = signal_table.copy()
    signal_df['datetime']=pd.to_datetime(signal_df['datetime'])
    signal_df['month']=signal_df['datetime'].dt.to_period("M")
    rebalance_dates=(
        signal_df.groupby("month",as_index=False)['datetime'].max().rename(columns={"datetime":"rebalance_dt"})
    )

    for dt in rebalance_dates['rebalance_dt']:
        day_df=signal_df[signal_df["datetime"]==dt]
        day_df=day_df.sort_values(["score","instrument"],ascending=[False,True])
        target_df=day_df[(day_df["score"]>0)].head(max_positions).copy()
        target_positions=set(target_df['instrument'].tolist())
        target_weight=1.0/float(len(target_positions)) if target_positions else 0
        score_map=day_df.set_index("instrument")['score'].to_dict()
        for instrument in sorted(set(positions)-target_positions):
            orders.append(
                {
                    "datetime": dt,
                    "instrument": instrument,
                    "action": -1,
                    "weight": positions[instrument],
                    "score": score_map.get(instrument,0),
                }
            )
        for instrument in sorted(target_positions-set(positions)):
            orders.append(
                {
                    "datetime": dt,
                    "instrument": instrument,
                    "action": 1,
                    "weight": target_weight,
                    "score": score_map.get(instrument, 0),
                }
            )
        positions={instrument:target_weight for instrument in target_positions}

    orders_df=pd.DataFrame(orders)
    # orders_df=orders_df.sort_values(["datetime","action","instrument"],ascending=[True,True,True]).reset_index(drop=True)

    orders_df=orders_df.round(3)
    return orders_df




def load_open_prices(instruments: list, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    opens = D.features(
        instruments=sorted(set(instruments)),
        fields=["$open"],
        start_time=start_dt.strftime("%Y-%m-%d"),
        end_time=end_dt.strftime("%Y-%m-%d"),
        freq="day",
    ).reset_index()
    opens.rename(columns={"$open": "open"}, inplace=True)
    opens["datetime"] = pd.to_datetime(opens["datetime"])
    open_table = opens.pivot_table(index="datetime", columns="instrument", values="open", aggfunc="last")
    return open_table


def build_optimal_check_detail(signal_table: pd.DataFrame, open_table: pd.DataFrame, max_positions: int = 4) -> pd.DataFrame:
    signal_df = signal_table.copy()
    signal_df["datetime"] = pd.to_datetime(signal_df["datetime"])
    signal_df["month"] = signal_df["datetime"].dt.to_period("M")
    rebalance_dates = (
        signal_df.groupby("month", as_index=False)["datetime"].max().sort_values("datetime")["datetime"].tolist()
    )
    if len(rebalance_dates) < 2:
        return pd.DataFrame(
            columns=[
                "rebalance_date",
                "next_rebalance_date",
                "entry_open_date",
                "exit_open_date",
                "selected_4",
                "optimal_4",
                "selected_avg_return",
                "optimal_avg_return",
                "return_gap",
                "overlap_count",
                "is_optimal",
            ]
        )

    records = []
    trade_dates = open_table.index.sort_values()

    def next_trade_date(dt: pd.Timestamp):
        pos = trade_dates.searchsorted(dt, side="right")
        if pos >= len(trade_dates):
            return None
        return trade_dates[pos]

    for i in range(len(rebalance_dates) - 1):
        dt = pd.to_datetime(rebalance_dates[i])
        next_dt = pd.to_datetime(rebalance_dates[i + 1])
        day_df = signal_df[signal_df["datetime"] == dt].sort_values(["score", "instrument"], ascending=[False, True])
        selected_list = day_df[(day_df["score"] > 0)].head(max_positions)["instrument"].tolist()

        entry_dt = next_trade_date(dt)
        exit_dt = next_trade_date(next_dt)
        if entry_dt is None or exit_dt is None:
            continue

        pool = day_df["instrument"].dropna().tolist()
        start_open = open_table.loc[entry_dt]
        end_open = open_table.loc[exit_dt]
        ret_map = {}
        for inst in pool:
            p0 = start_open.get(inst, np.nan)
            p1 = end_open.get(inst, np.nan)
            if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                ret_map[inst] = float(p1 / p0 - 1.0)

        ranked = sorted(ret_map.items(), key=lambda x: (-x[1], x[0]))
        optimal_list = [inst for inst, _ in ranked[:max_positions]]
        selected_rets = [ret_map[inst] for inst in selected_list if inst in ret_map]
        optimal_rets = [ret_map[inst] for inst in optimal_list if inst in ret_map]
        selected_avg = float(np.mean(selected_rets)) if selected_rets else np.nan
        optimal_avg = float(np.mean(optimal_rets)) if optimal_rets else np.nan
        overlap_count = len(set(selected_list) & set(optimal_list))
        is_optimal = (
            len(selected_list) == max_positions
            and len(optimal_list) == max_positions
            and set(selected_list) == set(optimal_list)
        )

        records.append(
            {
                "rebalance_date": dt.strftime("%Y-%m-%d"),
                "next_rebalance_date": next_dt.strftime("%Y-%m-%d"),
                "entry_open_date": entry_dt.strftime("%Y-%m-%d"),
                "exit_open_date": exit_dt.strftime("%Y-%m-%d"),
                "selected_4": "|".join(selected_list),
                "optimal_4": "|".join(optimal_list),
                "selected_avg_return": selected_avg,
                "optimal_avg_return": optimal_avg,
                "return_gap": selected_avg - optimal_avg if pd.notna(selected_avg) and pd.notna(optimal_avg) else np.nan,
                "overlap_count": overlap_count,
                "is_optimal": int(is_optimal),
            }
        )

    detail_df = pd.DataFrame(records)
    if detail_df.empty:
        return detail_df
    for c in ["selected_avg_return", "optimal_avg_return", "return_gap"]:
        detail_df[c] = pd.to_numeric(detail_df[c], errors="coerce").round(6)
    return detail_df


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default=str(script_dir / "input"))
    parser.add_argument("--calendar-path", type=str, default="/home/quant/zc/finance_deal/qlib_data/price_data0821/calendars/day.txt")
    parser.add_argument("--output", type=str, default=str(script_dir / "trade_portfolio_24_1.csv"))
    parser.add_argument("--max_positions", type=int, default=3)
    parser.add_argument("--single-output-name",type=str,default="trade_signal_24_1.csv")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    calendar_path = Path(args.calendar_path) if args.calendar_path else None
    output_path = Path(args.output)

    # stock_map = {
    #     "水井坊": "600779.SH",
    #     "五粮液": "000858.SZ",
    #     "贵州茅台": "600519.SH",
    #     "今世缘": "603369.SH",
    #     "泸州老窖": "000568.SZ",
    #     "古井贡酒": "000596.SZ",
    #     "口子窖": "603589.SH",
    #     "老白干酒": "600559.SH",
    #     "山西汾酒": "600809.SH",
    #     "迎驾贡酒": "603198.SH",
    #     "洋河股份": "002304.SZ",
    #     "伊力特": "600197.SH",
    # }
    stock_map={
        "比亚迪":"002594.SZ",
        "宁德时代": "300750.SZ",
        "欣旺达": "300207.SZ",
        "亿纬锂能": "300014.SZ",
        "珠海冠宇": "688772.SH",
        "孚能科技": "688567.SH",
    }
    fit_score_w = True
    calendar_df = build_calendar(input_root, calendar_path)
    signal_tables = []
    for csv_file in sorted(input_root.glob("*/back_tests_winloss.csv")):
        folder_name = csv_file.parent.name
        if folder_name in stock_map and fit_score_w:
            instrument_name=stock_map.get(folder_name,folder_name)
            # single_signal_table=build_single_signal_table(csv_file, instrument_name, calendar_df)

            feature_table = build_single_feature_table(csv_file, instrument_name, calendar_df)
            open_table = load_open_prices(feature_table['instrument'].unique().tolist(),
                                          start_dt=feature_table['datetime'].min(),
                                          end_dt=feature_table['datetime'].max() + pd.offsets.MonthEnd(1))
            samples_df = build_forward_return_samples(feature_table, open_table)
            fit_result = fit_score_weights_linear_regression(samples_df, pd.to_datetime("2024-10-30"))
            score_w = fit_result['weights']
            test_pred=fit_result['test_pred']
            print(score_w.round(2))
            single_signal_table = build_single_signal_table(csv_file, instrument_name, calendar_df, score_w=score_w,test_pred=test_pred)

            # single_signal_table=build_single_signal_table_up_valley(csv_file, instrument_name, calendar_df)
            single_output_path=csv_file.parent/args.single_output_name
            single_signal_table.to_csv(single_output_path,index=False)
            signal_tables.append(single_signal_table)




    if not signal_tables:
        raise ValueError(f"未在目录中找到 back_tests.csv: {input_root}")

    multi_signals = pd.concat(signal_tables, ignore_index=True)
    multi_signals = multi_signals.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    portfolio_orders = build_portfolio_orders(multi_signals,max_positions=args.max_positions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_orders.to_csv(output_path, index=False)
    print(f"signals={len(multi_signals)}, orders={len(portfolio_orders)}")
    print(f"saved: {output_path}")

    all_dates=pd.to_datetime(multi_signals['datetime'])
    open_table=load_open_prices(multi_signals['instrument'].unique().tolist(),start_dt=all_dates.min(),end_dt=all_dates.max()+pd.offsets.MonthEnd(1))
    optimal_check_df=build_optimal_check_detail(signal_table=multi_signals,open_table=open_table,max_positions=args.max_positions)
    print(optimal_check_df)
    optimal_check_df.to_csv(f"optimal_白酒.csv",index=False,encoding='utf-8-sig')


if __name__ == "__main__":
    main()



