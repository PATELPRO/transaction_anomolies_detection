import pandas as pd
import numpy as np
import yfinance as yf


def fetch_stock_data(ticker, period="3y", interval="1d"):

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True
    )

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.DatetimeIndex(
        raw.index).tz_localize(None).normalize()

    raw = raw[raw['Volume'] > 0]

    log_vol = np.log(raw['Volume'])
    log_vol.name = 'log_volume'
    log_vol = log_vol.asfreq('B').ffill()

    return log_vol


def fetch_latest_price(ticker):

    raw = yf.download(
        ticker,
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if raw.empty:
        return {'price': None, 'date': None}

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.DatetimeIndex(
        raw.index).tz_localize(None).normalize()
    raw = raw[raw['Volume'] > 0]

    return {
        'price': round(float(raw['Close'].iloc[-1]), 2),
        'date' : raw.index[-1].date()
    }