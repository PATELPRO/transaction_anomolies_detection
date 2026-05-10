import pandas as pd
import numpy as np
import calendar
from statsmodels.tsa.seasonal import STL


# ── expiry date logic ──────────────────────────────────────

REGIME_CHANGE_DATE = pd.Timestamp('2025-09-01')


def _get_last_weekday_of_month(year: int,
                                month: int,
                                weekday: int) -> pd.Timestamp:
    """
    Return the last occurrence of weekday in a month.
    weekday: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
    """
    last_day = calendar.monthrange(year, month)[1]
    for day in range(last_day, last_day - 7, -1):
        if calendar.weekday(year, month, day) == weekday:
            return pd.Timestamp(year, month, day)
    return None


def build_expiry_set(index: pd.DatetimeIndex) -> set:
    """
    Build set of NSE monthly F&O expiry dates
    covering the full range of the given index.

    Rule:
      Before Sep 1 2025 → last Thursday of month
      After  Sep 1 2025 → last Tuesday of month
    """
    expiry_dates = set()

    for year in range(index[0].year, index[-1].year + 1):
        for month in range(1, 13):
            month_start = pd.Timestamp(year, month, 1)
            weekday     = (3 if month_start < REGIME_CHANGE_DATE
                           else 1)
            exp = _get_last_weekday_of_month(
                year, month, weekday)
            if exp is not None:
                expiry_dates.add(exp)

    return expiry_dates


# ── core scoring pipeline ──────────────────────────────────

def compute_anomaly_scores(
        log_vol: pd.Series,
        window: int = 60,
        stl_period: int = 5) -> pd.DataFrame:
    """
    Run full anomaly detection pipeline on log volume series.

    Steps:
      1. STL decomposition (period=5 for weekly seasonality)
      2. Rolling z-score on STL residuals
      3. Rolling z-score on day-over-day volume shocks
      4. Combined weighted anomaly score

    Parameters
    ----------
    log_vol    : log volume series with business day frequency
    window     : rolling window in trading days (default 60 = 3 months)
    stl_period : seasonal period (default 5 = weekly)

    Returns
    -------
    pd.DataFrame with columns:
      log_volume, stl_residual, z_stl, z_shock, anomaly_score
    """
    # stl decomposition
    stl    = STL(log_vol, period=stl_period, robust=True)
    result = stl.fit()

    residual = pd.Series(result.resid,
                         index=log_vol.index,
                         name='stl_residual')

    # signal 1 — stl residual z-score
    roll_mean = residual.rolling(window).mean()
    roll_std  = residual.rolling(window).std()
    z_stl     = (residual - roll_mean) / roll_std
    z_stl.name = 'z_stl'

    # signal 2 — volume shock z-score
    shock      = log_vol.diff()
    shock_mean = shock.rolling(window).mean()
    shock_std  = shock.rolling(window).std()
    z_shock    = (shock - shock_mean) / shock_std
    z_shock.name = 'z_shock'

    # combined score
    anomaly_score = (0.6 * z_stl.abs().fillna(0) +
                     0.4 * z_shock.abs().fillna(0))
    anomaly_score.name = 'anomaly_score'

    return pd.DataFrame({
        'log_volume'    : log_vol,
        'stl_residual'  : residual,
        'z_stl'         : z_stl,
        'z_shock'       : z_shock,
        'anomaly_score' : anomaly_score,
    })


def classify_scores(
        scored: pd.DataFrame,
        warning_threshold: float  = 2.0,
        critical_threshold: float = 3.0,
        expiry_warning: float     = 3.0,
        expiry_critical: float    = 4.0) -> pd.Series:
    """
    Classify each day as normal / warning / critical.
    Uses expiry-aware thresholds — higher on expiry days
    because elevated volume is expected and normal.

    Returns pd.Series of labels aligned to scored.index
    """
    expiry_set = build_expiry_set(scored.index)

    labels = []
    for date, score in scored['anomaly_score'].items():
        is_expiry = date in expiry_set
        warn_t    = expiry_warning  if is_expiry else warning_threshold
        crit_t    = expiry_critical if is_expiry else critical_threshold

        if score >= crit_t:
            labels.append('critical')
        elif score >= warn_t:
            labels.append('warning')
        else:
            labels.append('normal')

    return pd.Series(labels,
                     index=scored.index,
                     name='label')


def get_anomaly_log(scored: pd.DataFrame,
                    labels: pd.Series) -> pd.DataFrame:
    """
    Return clean table of all flagged observations
    sorted by anomaly score descending.
    """
    flagged = scored[labels != 'normal'].copy()
    flagged['label'] = labels[labels != 'normal']
    flagged = flagged.sort_values(
        'anomaly_score', ascending=False)
    flagged = flagged.round(3)
    return flagged