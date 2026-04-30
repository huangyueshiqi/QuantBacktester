import numpy as np


def _to_1d_float_array(x):
    if x is None:
        return np.asarray([], dtype="float64")
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy(dtype="float64", copy=False)
    else:
        arr = np.asarray(x, dtype="float64")
    arr = np.asarray(arr, dtype="float64").reshape(-1)
    return arr[~np.isnan(arr)]


def _annualization(annualization):
    return 252 if annualization is None else int(annualization)


class stats:
    @staticmethod
    def cum_returns_final(returns):
        r = _to_1d_float_array(returns)
        if r.size == 0:
            return 0.0
        return float(np.prod(1.0 + r) - 1.0)

    @staticmethod
    def max_drawdown(returns):
        r = _to_1d_float_array(returns)
        if r.size == 0:
            return 0.0
        cum = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        dd = cum / peak - 1.0
        return float(np.min(dd))

    @staticmethod
    def annual_return(returns, annualization=252):
        r = _to_1d_float_array(returns)
        if r.size == 0:
            return 0.0
        ann = _annualization(annualization)
        total = np.prod(1.0 + r)
        return float(total ** (ann / r.size) - 1.0)

    @staticmethod
    def annual_volatility(returns, annualization=252):
        r = _to_1d_float_array(returns)
        if r.size < 2:
            return 0.0
        ann = _annualization(annualization)
        return float(np.std(r, ddof=1) * np.sqrt(ann))

    @staticmethod
    def sharpe_ratio(returns, risk_free=0.0, annualization=252):
        r = _to_1d_float_array(returns)
        if r.size < 2:
            return 0.0
        ann = _annualization(annualization)
        ex = r - float(risk_free)
        vol = np.std(ex, ddof=1)
        if vol == 0 or np.isnan(vol):
            return 0.0
        return float(np.mean(ex) / vol * np.sqrt(ann))

    @staticmethod
    def sortino_ratio(returns, required_return=0.0, annualization=252):
        r = _to_1d_float_array(returns)
        if r.size < 2:
            return 0.0
        ann = _annualization(annualization)
        req = float(required_return)
        ex = r - req
        downside = np.minimum(ex, 0.0)
        dd = np.std(downside, ddof=1)
        if dd == 0 or np.isnan(dd):
            return 0.0
        return float(np.mean(ex) * ann / (dd * np.sqrt(ann)))

    @staticmethod
    def omega_ratio(returns, risk_free=0.0):
        r = _to_1d_float_array(returns)
        if r.size == 0:
            return 0.0
        thr = float(risk_free)
        gains = np.sum(np.maximum(r - thr, 0.0))
        losses = np.sum(np.maximum(thr - r, 0.0))
        if losses == 0 or np.isnan(losses):
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    @staticmethod
    def calmar_ratio(returns, annualization=252):
        mdd = stats.max_drawdown(returns)
        if mdd >= 0:
            return 0.0
        ar = stats.annual_return(returns, annualization=annualization)
        denom = abs(mdd)
        return float(ar / denom) if denom > 0 else 0.0

    @staticmethod
    def beta(returns, factor_returns, risk_free=0.0):
        r = _to_1d_float_array(returns)
        f = _to_1d_float_array(factor_returns)
        n = min(r.size, f.size)
        if n < 2:
            return 0.0
        rf = float(risk_free)
        r = r[:n] - rf
        f = f[:n] - rf
        var = np.var(f, ddof=1)
        if var == 0 or np.isnan(var):
            return 0.0
        cov = np.cov(r, f, ddof=1)[0, 1]
        return float(cov / var)

    @staticmethod
    def alpha(returns, factor_returns, risk_free=0.0, annualization=252):
        r = _to_1d_float_array(returns)
        f = _to_1d_float_array(factor_returns)
        n = min(r.size, f.size)
        if n < 2:
            return 0.0
        ann = _annualization(annualization)
        rf = float(risk_free)
        r = r[:n]
        f = f[:n]
        b = stats.beta(r, f, risk_free=rf)
        ex_port = r - rf
        ex_fact = f - rf
        a_daily = np.mean(ex_port - b * ex_fact)
        return float(a_daily * ann)


try:
    import empyrical as _empyrical

    stats = _empyrical.stats
except Exception:
    pass

