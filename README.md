---
title: Transaction Anomaly Detection
emoji: 📊
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---
# Real-Time Financial Anomaly Detection Engine
### Statistical Detection of Anomalous Market Activity in Indian Banking Stocks

[![Streamlit App](https://img.shields.io/badge/Live%20App-Hugging%20Face-blue)](https://huggingface.co/spaces/PATELPRO/transaction-anomaly-detection)

[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Live Demo

🔗 **[Open the App](https://huggingface.co/spaces/PATELPRO/transaction-anomaly-detection)**

The dashboard monitors Stocks (Ex - NSE: HDFCBANK) daily trading volume for statistically unusual behavior. It pulls live data from NSE via Yahoo Finance, runs the full anomaly detection pipeline, and surfaces flagged dates with anomaly scores and classification.

---
#### NOTE: AI SUMMARY
## Project Overview

This project started as a **time series forecasting** problem and evolved — through rigorous empirical analysis — into a **distribution-based anomaly detection** system. The journey from forecasting to anomaly detection is documented honestly across six notebooks and is itself one of the most important outcomes of the project.

### The Original Goal
Forecast future trading volume using classical time series models (ARIMA, SARIMA), then flag deviations from the forecast as anomalies. This is the standard approach in the time series anomaly detection literature.

### What the Data Told Us
HDFC Bank daily volume approximates a **random walk** — consistent with weak-form market efficiency (Fama, 1970). Every model from naive baseline to fully specified SARIMA produced statistically equivalent forecasts. White noise residuals confirmed that no exploitable autocorrelation structure exists in the series at daily frequency.

### The Methodology Shift
Forecasting-based anomaly detection was replaced with **distribution-based detection** — which does not require the series to be predictable, only that the distribution of normal behavior is stable.

---

## Repository Structure
transaction-anomaly-detection/
│
├── notebooks/
│   ├── 01_data_ingestion_eda.ipynb
│   ├── 02_time_series_analysis.ipynb
│   ├── 03_forecasting_engine.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_scalability_extensions.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── anomaly_detection.py
│   ├── preprocessing.py
│   ├── forecasting.py
│   └── evaluation.py
│
├── app/
│   ├── main.py
│   └── components/
│
├── config/
│   └── config.yaml
│
├── Dockerfile
└── requirements.txt
---

## The Full Project Journey

### Notebook 1 — Data Ingestion and EDA

**Data source:** NSE via Yahoo Finance (`yfinance`) — 3 years of daily OHLCV data for 8 Indian banking and fintech stocks (HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK, SBIN, BAJFINANCE, PAYTM, HDFCLIFE) and two indices (Nifty 50, Nifty Bank).

**Why volume?**
Trading volume on NSE was used as a proxy for financial transaction activity. Both exhibit identical time series properties — intraday seasonality, weekly patterns, event-driven spikes, and regime changes. This approach mirrors how production anomaly detection systems are developed before access to proprietary transaction data is available.

**Key findings from EDA:**
- Strong weekly seasonality confirmed — volume varies systematically by day of week
- Monthly patterns present — February (Union Budget), March (FY end), October (festive season) show elevated activity
- Log transformation justified — raw volume is right-skewed; log volume is approximately normal

---

### Notebook 2 — Time Series Analysis

**Decomposition:**
STL (Seasonal-Trend decomposition using Loess) was applied with a weekly period (s=5 trading days). The robust=True parameter was used to handle outliers. The residual component — what remains after removing trend and seasonality — forms the primary anomaly signal.

**Stationarity testing:**
- ADF and KPSS tests applied to raw volume, log volume, and first-differenced log volume
- Raw volume and log volume: non-stationary (confirmed by both tests)
- First-differenced log volume: stationary (confirmed by both tests)
- Conclusion: d=1 required for ARIMA/SARIMA

**ACF/PACF analysis on first-differenced log volume:**
- ACF cuts off sharply at lag 1 → suggests MA(1) non-seasonal component
- Significant ACF spikes at lags 5 and 10 (multiples of s=5) → confirms weekly seasonality
- Seasonal ACF cuts off after lag 10 → suggests seasonal MA(1) (Q=1)
- PACF not significant at seasonal lags → P=0

**Parameter identification:**
p=0, d=1, q=1  (non-seasonal)
P=0, D=0, Q=1, s=5  (seasonal)
Primary candidate: SARIMA(0,1,1)(0,0,1)[5]

---

### Notebook 3 — Forecasting Engine and EMH Finding

**Ten models built and evaluated** on the same 80/20 train/test split (walk-forward, no shuffling):

| Model | Description |
|-------|-------------|
| Naive (Random Walk) | Forecast = last observed value |
| Seasonal Naive | Forecast = same day last week |
| AR(1) | Short-term persistence |
| MA(1) | Short-term shock propagation |
| ARMA(1,1) | Persistence and shocks combined |
| ARIMA(0,1,1) | MA with differencing |
| ARIMA(1,1,1) | AR+MA with differencing |
| SARIMA(0,1,1)(0,0,1)[5] | Primary candidate |
| SARIMA(1,1,1)(0,0,1)[5] | Extended SARIMA |
| SARIMA(1,1,1)(1,0,1)[5] | Full SARIMA |

**Key finding:**
Every model approximated the naive random walk baseline. The best SARIMA showed marginal trend but achieved worse RMSE than naive. Ljung-Box testing on residuals confirmed white noise — meaning the model extracted everything extractable and what remains is genuinely random.

**Empirical conclusion:**
HDFC Bank daily volume exhibits **weak-form market efficiency** (Fama, 1970). Past volume data contains no exploitable predictive information at daily frequency. This is expected for one of the most liquid stocks on NSE.

**Why the random walk is not a modeling failure:**
A model whose residuals are white noise is correctly reporting that the series has no remaining predictable structure. Models with more parameters (SARIMA(1,1,1)(1,0,1)[5]) partially overfit the noise and performed worse out of sample — the classic signature of fitting a random series.

**Documented methodology decision:**
Since SARIMA confidence intervals inherit the unreliability of the point forecast, using them as anomaly detection boundaries was abandoned. Distribution-based detection was adopted instead.

---

### Notebook 4 — Anomaly Detection Engine

**Core insight:**
Anomaly detection does not require forecasting. It requires a stable model of normal behavior. Even if tomorrow's volume is unpredictable, a 5x volume spike is detectable as statistically inconsistent with the historical distribution of normal daily behavior.

**Two complementary signals:**

**Signal 1 — STL Residual Z-Score (weight: 0.6)**
STL decomposition separates volume into trend + seasonality + residual
Residual = what cannot be explained by normal seasonal and trend behavior
Rolling z-score computed on residuals (60-day window)
Large z-score = volume is unusually high or low relative to seasonal expectation

**Signal 2 — Volume Shock Z-Score (weight: 0.4)**
Daily log volume change computed
Rolling z-score on changes (60-day window)
Large z-score = unusually large single-day move
Captures sudden spikes regardless of seasonal context

**Combined anomaly score:**
anomaly_score = 0.6 × |z_stl| + 0.4 × |z_shock|

**Expiry-aware thresholds:**

NSE F&O monthly expiry days (last Thursday before Sep 2025, last Tuesday after Sep 2025) use elevated thresholds because high volume is expected and normal on settlement days. This reduces predictable false alarms.
Standard days:  Warning ≥ 2.0  |  Critical ≥ 3.0
Expiry days:    Warning ≥ 3.0  |  Critical ≥ 4.0

---

### Notebook 5 — Evaluation

**Why not precision/recall?**
Standard precision/recall evaluation requires a complete, verified set of ground truth labels. A manually curated event list is incomplete and biased toward events we already knew about. Building metrics on top of it would create a false impression of rigor.

**Qualitative case study:**
Top 10 anomalies were surfaced and researched. The majority of high-scoring dates corresponded to identifiable real-world events in HDFC Bank or the broader Indian banking sector. Events with strong backing included HDFC-HDFC Bank merger completion, quarterly earnings surprises, and RBI MPC decisions.

**Anomaly persistence analysis:**
For each critical anomaly, the anomaly score was tracked for 5 trading days before and after. Real market events create multi-day volume ripples — the average score profile showed the expected decay pattern, confirming the system is detecting genuine events rather than statistical noise.

**Honest limitations:**
- Ground truth is incomplete — some real anomalies are certainly missed
- Single stock — HDFC Bank's liquidity dampens macro event signal

---

### Notebook 6 — SARIMAX Extension

**Motivation:**
If the forecasting failure was due to missing external information rather than genuine randomness, adding exogenous variables (making it SARIMAX) might restore predictive power.

**Three features tested:**

| Feature | Rationale |
|---------|-----------|
| NSE F&O expiry dummy | Last Thursday/Tuesday of month — forced position closing creates predictable volume |
| India VIX (lagged 1 day) | Fear index — higher VIX predicts elevated next-day trading activity |
| Nifty Bank return (lagged 1 day, absolute) | Large sector moves predict elevated next-day volume |

**Result:**
SARIMAX with all three features did not meaningfully improve over the naive baseline despite statistically significant coefficients. This is the third line of evidence confirming the distribution-based approach is correct.
---

## Anomaly Detection System Architecture

NSE Daily Volume Data (yfinance)
↓
Log Transformation
↓
Business Day Frequency Alignment
↓
STL Decomposition (period=5)
↙              ↘
Trend + Seasonal    Residual
(removed)           ↓
Rolling Z-Score
(60-day window)
↓
STL Residual Signal (60%)
+
Volume Shock Signal (40%)
↓
Combined Anomaly Score
↓
Expiry-Aware Classification
Normal | Warning | Critical

---