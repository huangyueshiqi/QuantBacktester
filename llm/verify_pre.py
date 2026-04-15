from pathlib import Path
import argparse
import math
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


SIGNAL_COLS = ["上升区间", "下降区间", "峰点", "谷点"]
DEFAULT_STOCK_MAP = {
    "五粮液": "000858.SZ",
    "贵州茅台": "600519.SH",
    "今世缘": "603369.SH",
    "泸州老窖": "000568.SZ",
    "古井贡酒": "000596.SZ",
    "口子窖": "603589.SH",
    "老白干酒": "600559.SH",
    "山西汾酒": "600809.SH",
    "水井坊": "600779.SH",
    "迎驾贡酒": "603198.SH",
    "洋河股份": "002304.SZ",
    "伊力特": "600197.SH",
}


def parse_horizons(raw: str) -> List[int]:
    horizons = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v > 0:
            horizons.append(v)
    horizons = sorted(set(horizons))
    if not horizons:
        raise ValueError("horizons 不能为空")
    return horizons


def normal_pvalue_from_t(t_value: float) -> float:
    z = abs(float(t_value))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, 2.0 * (1.0 - cdf)))


def summarize_returns(ret: np.ndarray, bootstrap_rounds: int, rng: np.random.Generator) -> Dict[str, float]:
    n = int(ret.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    mean_v = float(np.mean(ret))
    med_v = float(np.median(ret))
    win_v = float(np.mean(ret > 0))
    std_v = float(np.std(ret, ddof=1)) if n > 1 else np.nan
    if n > 1 and np.isfinite(std_v) and std_v > 0:
        t_stat = mean_v / (std_v / math.sqrt(n))
        p_value = normal_pvalue_from_t(t_stat)
    else:
        t_stat = np.nan
        p_value = np.nan

    if n > 1 and bootstrap_rounds > 0:
        idx = rng.integers(0, n, size=(bootstrap_rounds, n))
        boot_means = ret[idx].mean(axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        ci_low = float(ci_low)
        ci_high = float(ci_high)
    else:
        ci_low, ci_high = np.nan, np.nan

    return {
        "n": n,
        "mean": mean_v,
        "median": med_v,
        "win_rate": win_v,
        "std": std_v,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def compute_max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    return float(dd.min())


def build_month_last_trade_map(price_df: pd.DataFrame) -> Dict[pd.Period, pd.Timestamp]:
    temp = price_df[["datetime"]].copy()
    temp["month"] = temp["datetime"].dt.to_period("M")
    grouped = temp.groupby("month", as_index=False)["datetime"].max()
    return {row["month"]: row["datetime"] for _, row in grouped.iterrows()}


def load_signal_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "日期" not in df.columns:
        raise ValueError(f"文件缺少 日期 列: {csv_path}")
    for c in SIGNAL_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    for c in SIGNAL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df[["日期"] + SIGNAL_COLS]


def align_events_to_trade_days(signal_df: pd.DataFrame, month_last_map: Dict[pd.Period, pd.Timestamp], threshold: float) -> pd.DataFrame:
    temp = signal_df.copy()
    temp["month"] = temp["日期"].dt.to_period("M")
    temp["event_dt"] = temp["month"].map(month_last_map)
    temp = temp.dropna(subset=["event_dt"])
    event_rows = []
    for _, row in temp.iterrows():
        for sig in SIGNAL_COLS:
            val = float(row[sig])
            if val > threshold:
                event_rows.append(
                    {
                        "signal_name": sig,
                        "signal_value": val,
                        "signal_dt": row["日期"],
                        "event_dt": row["event_dt"],
                    }
                )
    return pd.DataFrame(event_rows)


def evaluate_single_stock(
    folder_name: str,
    instrument: str,
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizons: List[int],
    hit_horizon: int,
    extremum_window: int,
    threshold: float,
    bootstrap_rounds: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_last_map = build_month_last_trade_map(price_df)
    events = align_events_to_trade_days(signal_df, month_last_map, threshold=threshold)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    px = price_df[["datetime", "close"]].dropna().sort_values("datetime").reset_index(drop=True)
    px["idx"] = np.arange(len(px))
    idx_map = px.set_index("datetime")["idx"].to_dict()
    close_np = px["close"].to_numpy()

    event_ret_rows = []
    hit_rows = []
    monthly_bt_rows = []

    for sig_name, grp in events.groupby("signal_name"):
        event_indices = [idx_map.get(dt) for dt in grp["event_dt"]]
        event_indices = [x for x in event_indices if x is not None]
        event_indices = np.array(event_indices, dtype=int)
        if event_indices.size == 0:
            continue

        for h in horizons:
            valid_mask = event_indices + h < len(close_np)
            valid_idx = event_indices[valid_mask]
            if valid_idx.size == 0:
                stats = summarize_returns(np.array([], dtype=float), bootstrap_rounds, rng)
            else:
                ret = close_np[valid_idx + h] / close_np[valid_idx] - 1.0
                stats = summarize_returns(ret.astype(float), bootstrap_rounds, rng)
            event_ret_rows.append(
                {
                    "name": folder_name,
                    "instrument": instrument,
                    "signal_name": sig_name,
                    "horizon_days": h,
                    **stats,
                }
            )

        valid_mask_hit = event_indices + hit_horizon < len(close_np)
        valid_idx_hit = event_indices[valid_mask_hit]
        if valid_idx_hit.size > 0:
            hit_ret = close_np[valid_idx_hit + hit_horizon] / close_np[valid_idx_hit] - 1.0
            if sig_name == "上升区间":
                hit_flag = hit_ret > 0
            elif sig_name == "下降区间":
                hit_flag = hit_ret < 0
            else:
                hit_flag = np.zeros_like(hit_ret, dtype=bool)
        else:
            hit_flag = np.array([], dtype=bool)

        if sig_name in ("峰点", "谷点"):
            ext_flags = []
            for idx in event_indices:
                left = max(0, idx - extremum_window)
                right = min(len(close_np) - 1, idx + extremum_window)
                window = close_np[left : right + 1]
                if window.size <= 1:
                    continue
                if sig_name == "峰点":
                    ext_flags.append(close_np[idx] >= np.max(window))
                else:
                    ext_flags.append(close_np[idx] <= np.min(window))
            ext_flags = np.array(ext_flags, dtype=bool)
            extremum_rate = float(np.mean(ext_flags)) if ext_flags.size > 0 else np.nan
            extremum_n = int(ext_flags.size)
        else:
            extremum_rate = np.nan
            extremum_n = 0

        hit_rows.append(
            {
                "name": folder_name,
                "instrument": instrument,
                "signal_name": sig_name,
                "direction_hit_horizon_days": hit_horizon,
                "direction_hit_n": int(hit_flag.size),
                "direction_hit_rate": float(np.mean(hit_flag)) if hit_flag.size > 0 else np.nan,
                "extremum_window_days": extremum_window,
                "extremum_hit_n": extremum_n,
                "extremum_hit_rate": extremum_rate,
            }
        )

    bt_df = signal_df.copy()
    bt_df["buy_score"] = bt_df["上升区间"] + bt_df["谷点"]
    bt_df["sell_score"] = bt_df["下降区间"] + bt_df["峰点"]
    bt_df["position"] = ((bt_df["buy_score"] > bt_df["sell_score"]) & (bt_df["buy_score"] > threshold)).astype(float)
    bt_df["month"] = bt_df["日期"].dt.to_period("M")
    month_close = price_df.copy()
    month_close["month"] = month_close["datetime"].dt.to_period("M")
    month_end_px = month_close.groupby("month", as_index=False).last()[["month", "close"]]
    bt_df = pd.merge(bt_df, month_end_px, on="month", how="inner").sort_values("month").reset_index(drop=True)
    bt_df["next_ret"] = bt_df["close"].shift(-1) / bt_df["close"] - 1.0
    bt_df["strategy_ret"] = bt_df["position"] * bt_df["next_ret"]
    bt_df = bt_df.dropna(subset=["strategy_ret"]).copy()
    if not bt_df.empty:
        nav = (1.0 + bt_df["strategy_ret"]).cumprod()
        ann_factor = 12.0
        avg_m = float(bt_df["strategy_ret"].mean())
        vol_m = float(bt_df["strategy_ret"].std(ddof=1)) if len(bt_df) > 1 else np.nan
        ann_ret = float((1.0 + avg_m) ** ann_factor - 1.0)
        ann_vol = float(vol_m * math.sqrt(ann_factor)) if np.isfinite(vol_m) else np.nan
        sharpe = float(ann_ret / ann_vol) if np.isfinite(ann_vol) and ann_vol > 0 else np.nan
        monthly_bt_rows.append(
            {
                "name": folder_name,
                "instrument": instrument,
                "months": int(len(bt_df)),
                "trade_months": int((bt_df["position"] > 0).sum()),
                "trade_ratio": float((bt_df["position"] > 0).mean()),
                "avg_monthly_ret": avg_m,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": compute_max_drawdown(nav),
                "monthly_win_rate": float((bt_df["strategy_ret"] > 0).mean()),
            }
        )

    return pd.DataFrame(event_ret_rows), pd.DataFrame(hit_rows), pd.DataFrame(monthly_bt_rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default=str(script_dir / "input"))
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "validation_output"))
    parser.add_argument("--provider-uri", type=str, default=os.environ.get("QLIB_PROVIDER_URI", "/home/quant/zc/finance_deal/qlib_data/price_data0821"))
    parser.add_argument("--horizons", type=str, default="1,3,5,10,20")
    parser.add_argument("--hit-horizon", type=int, default=5)
    parser.add_argument("--extremum-window", type=int, default=5)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    args = parser.parse_args()
    import qlib
    from qlib.data import D

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)

    csv_files = sorted(input_root.glob("*/back_tests_winloss.csv"))
    if not csv_files:
        raise ValueError(f"未在目录中找到 back_tests.csv: {input_root}")

    signals_by_stock = {}
    stock_list = []
    min_dt, max_dt = None, None
    for csv_file in csv_files:
        folder_name = csv_file.parent.name
        instrument = DEFAULT_STOCK_MAP.get(folder_name, folder_name)
        signal_df = load_signal_file(csv_file)
        signals_by_stock[(folder_name, instrument)] = signal_df
        stock_list.append(instrument)
        cur_min = signal_df["日期"].min()
        cur_max = signal_df["日期"].max()
        min_dt = cur_min if min_dt is None else min(min_dt, cur_min)
        max_dt = cur_max if max_dt is None else max(max_dt, cur_max)

    qlib.init(provider_uri=args.provider_uri)
    end_padding = max(max(horizons), args.hit_horizon, args.extremum_window) + 30
    prices = D.features(
        instruments=sorted(set(stock_list)),
        fields=["$close"],
        start_time=min_dt.strftime("%Y-%m-%d"),
        end_time=(max_dt + pd.Timedelta(days=end_padding)).strftime("%Y-%m-%d"),
        freq="day",
    ).reset_index()
    prices = prices.rename(columns={"$close": "close"})
    prices["datetime"] = pd.to_datetime(prices["datetime"])

    rng = np.random.default_rng(20260401)
    event_tables = []
    hit_tables = []
    backtest_tables = []
    for (folder_name, instrument), signal_df in signals_by_stock.items():
        price_df = prices[prices["instrument"] == instrument][["datetime", "close"]].dropna().sort_values("datetime")
        if price_df.empty:
            continue
        ev_df, hit_df, bt_df = evaluate_single_stock(
            folder_name=folder_name,
            instrument=instrument,
            signal_df=signal_df,
            price_df=price_df,
            horizons=horizons,
            hit_horizon=max(1, args.hit_horizon),
            extremum_window=max(1, args.extremum_window),
            threshold=args.signal_threshold,
            bootstrap_rounds=max(0, args.bootstrap_rounds),
            rng=rng,
        )
        if not ev_df.empty:
            event_tables.append(ev_df)
        if not hit_df.empty:
            hit_tables.append(hit_df)
        if not bt_df.empty:
            backtest_tables.append(bt_df)

    event_ret_df = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    hit_rate_df = pd.concat(hit_tables, ignore_index=True) if hit_tables else pd.DataFrame()
    backtest_df = pd.concat(backtest_tables, ignore_index=True) if backtest_tables else pd.DataFrame()

    if not event_ret_df.empty:
        agg = event_ret_df.groupby(["signal_name", "horizon_days"], as_index=False).agg(
            n=("n", "sum"),
            mean=("mean", "mean"),
            median=("median", "mean"),
            win_rate=("win_rate", "mean"),
            std=("std","mean"),
            t_stat=("t_stat", "mean"),
            p_value=("p_value", "mean"),
            ci_low=("ci_low", "mean"),
            ci_high=("ci_high", "mean"),
        )
        agg["name"] = "ALL"
        agg["instrument"] = "ALL"
        event_ret_df = pd.concat([event_ret_df, agg[event_ret_df.columns]], ignore_index=True)

    if not hit_rate_df.empty:
        agg = hit_rate_df.groupby(["signal_name"], as_index=False).agg(
            direction_hit_n=("direction_hit_n", "sum"),
            direction_hit_rate=("direction_hit_rate", "mean"),
            extremum_hit_n=("extremum_hit_n", "sum"),
            extremum_hit_rate=("extremum_hit_rate", "mean"),
        )
        agg["name"] = "ALL"
        agg["instrument"] = "ALL"
        agg["direction_hit_horizon_days"] = max(1, args.hit_horizon)
        agg["extremum_window_days"] = max(1, args.extremum_window)
        hit_rate_df = pd.concat([hit_rate_df, agg[hit_rate_df.columns]], ignore_index=True)

    if not backtest_df.empty:
        agg = pd.DataFrame(
            [
                {
                    "name": "ALL",
                    "instrument": "ALL",
                    "months": int(backtest_df["months"].sum()),
                    "trade_months": int(backtest_df["trade_months"].sum()),
                    "trade_ratio": float(backtest_df["trade_ratio"].mean()),
                    "avg_monthly_ret": float(backtest_df["avg_monthly_ret"].mean()),
                    "ann_return": float(backtest_df["ann_return"].mean()),
                    "ann_vol": float(backtest_df["ann_vol"].mean()),
                    "sharpe": float(backtest_df["sharpe"].mean()),
                    "max_drawdown": float(backtest_df["max_drawdown"].mean()),
                    "monthly_win_rate": float(backtest_df["monthly_win_rate"].mean()),
                }
            ]
        )
        backtest_df = pd.concat([backtest_df, agg], ignore_index=True)

    event_path = output_dir / "validation_event_returns.csv"
    hit_path = output_dir / "validation_hit_rates.csv"
    bt_path = output_dir / "validation_backtest_summary.csv"
    event_ret_df.to_csv(event_path, index=False, encoding="utf-8-sig")
    hit_rate_df.to_csv(hit_path, index=False, encoding="utf-8-sig")
    backtest_df.to_csv(bt_path, index=False, encoding="utf-8-sig")

    print(f"saved: {event_path}")
    print(f"saved: {hit_path}")
    print(f"saved: {bt_path}")
    print(
        f"stocks={len(signals_by_stock)}, event_rows={len(event_ret_df)}, "
        f"hit_rows={len(hit_rate_df)}, backtest_rows={len(backtest_df)}"
    )


if __name__ == "__main__":
    main()
