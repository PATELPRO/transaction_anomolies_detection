import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_ingestion import fetch_stock_data, fetch_latest_price
from src.anomaly_detection import (
    compute_anomaly_scores,
    classify_scores,
    get_anomaly_log
)

st.set_page_config(
    page_title="Stock Volume Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def load_data(ticker, period):
    log_vol = fetch_stock_data(ticker, period=period)
    scored  = compute_anomaly_scores(log_vol)
    labels  = classify_scores(scored)
    log     = get_anomaly_log(scored, labels)
    price   = fetch_latest_price(ticker)
    return log_vol, scored, labels, log, price


# ── sidebar ──
with st.sidebar:
    st.title("⚙️ Settings")

    ticker = st.selectbox(
        "Stock",
        ["HDFCBANK.NS", "ICICIBANK.NS",
         "AXISBANK.NS", "SBIN.NS"],
        index=0
    )

    date_range = st.selectbox(
        "Display period",
        ["Last 3 months", "Last 6 months",
         "Last 1 year", "Full history"],
        index=1
    )

    threshold_display = st.radio(
        "Show anomalies",
        ["Critical only", "Warning + Critical"],
        index=1
    )

    if st.button("🔄 Refresh data",
                 use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Data: NSE via Yahoo Finance  \n"
        "Refreshes daily at market close  \n"
        "Built with Streamlit"
    )


# ── load data ──
with st.spinner("Loading data..."):
    try:
        log_vol, scored, labels, anomaly_log, price = \
            load_data(ticker, period="3y")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()


# ── date filter ──
today     = scored.index[-1]
range_map = {
    "Last 3 months" : today - pd.tseries.offsets.BDay(63),
    "Last 6 months" : today - pd.tseries.offsets.BDay(126),
    "Last 1 year"   : today - pd.tseries.offsets.BDay(252),
    "Full history"  : scored.index[0],
}
start_date = range_map[date_range]

scored_filtered  = scored.loc[start_date:]
labels_filtered  = labels.loc[start_date:]
log_vol_filtered = log_vol.loc[start_date:]

if threshold_display == "Critical only":
    show_labels = ['critical']
else:
    show_labels = ['warning', 'critical']


# ── Section 1 — Status ──
st.title("Stock Volume Anomaly Monitor")
st.caption(
    f"NSE daily trading volume — "
    f"last updated {today.strftime('%d %b %Y')}"
)

latest_score = scored['anomaly_score'].iloc[-1]
latest_label = labels.iloc[-1]

flagged_dates = labels[labels.isin(['critical', 'warning'])]
last_anomaly  = (flagged_dates.index[-1]
                 if len(flagged_dates) > 0 else None)

status_color = {
    'normal'  : '🟢',
    'warning' : '🟡',
    'critical': '🔴'
}

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Today's Status",
        value=f"{status_color[latest_label]} "
              f"{latest_label.upper()}",
        delta=f"Score: {latest_score:.2f}"
    )

with col2:
    st.metric(
        label="Current Price",
        value=(f"₹{price['price']:,.2f}"
               if price['price'] else "N/A")
    )

with col3:
    st.metric(
        label="Last Anomaly",
        value=(last_anomaly.strftime('%d %b %Y')
               if last_anomaly else "None detected")
    )

with col4:
    n_critical = (labels == 'critical').sum()
    n_warning  = (labels == 'warning').sum()
    st.metric(
        label="Anomalies (3 years)",
        value=f"{n_critical} critical",
        delta=f"{n_warning} warnings"
    )

st.divider()


# ── Section 2 — Timeline ──
st.subheader("Volume Timeline")

critical_mask = labels_filtered == 'critical'
warning_mask  = labels_filtered == 'warning'

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.06
)

fig.add_trace(
    go.Scatter(
        x=log_vol_filtered.index,
        y=log_vol_filtered.values,
        mode='lines',
        name='Log Volume',
        line=dict(color='#adb5bd', width=1),
        customdata=np.column_stack([
            scored_filtered['anomaly_score'].values,
            labels_filtered.values
        ]),
        hovertemplate=(
            '<b>%{x|%d %b %Y}</b><br>'
            'Log Volume: %{y:.3f}<br>'
            'Anomaly Score: %{customdata[0]:.3f}<br>'
            'Status: %{customdata[1]}<br>'
            '<extra></extra>'
        )
    ),
    row=1, col=1
)

if 'warning' in show_labels and warning_mask.any():
    fig.add_trace(
        go.Scatter(
            x=log_vol_filtered[warning_mask].index,
            y=log_vol_filtered[warning_mask].values,
            mode='markers',
            name='Warning',
            marker=dict(color='#ffc107', size=7,
                        line=dict(width=0)),
            hovertemplate=None,
            hoverinfo='none'
        ),
        row=1, col=1
    )

if 'critical' in show_labels and critical_mask.any():
    fig.add_trace(
        go.Scatter(
            x=log_vol_filtered[critical_mask].index,
            y=log_vol_filtered[critical_mask].values,
            mode='markers',
            name='Critical',
            marker=dict(color='#dc3545', size=10,
                        line=dict(width=0)),
            hovertemplate=None,
            hoverinfo='none'
        ),
        row=1, col=1
    )

fig.add_trace(
    go.Scatter(
        x=scored_filtered.index,
        y=scored_filtered['anomaly_score'],
        mode='lines',
        name='Anomaly Score',
        line=dict(color='#6c757d', width=1),
        hovertemplate=(
            '%{x|%d %b %Y}<br>'
            'Score: %{y:.3f}<extra></extra>'
        )
    ),
    row=2, col=1
)

fig.add_hline(y=3.0, line_dash='dash',
              line_color='#dc3545',
              line_width=1, row=2, col=1)
fig.add_hline(y=2.0, line_dash='dash',
              line_color='#ffc107',
              line_width=1, row=2, col=1)

fig.update_layout(
    height=520,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=1.02,
        xanchor='left', x=0
    ),
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white',
)
fig.update_xaxes(showgrid=True,
                 gridcolor='#f0f0f0', gridwidth=1)
fig.update_yaxes(showgrid=True,
                 gridcolor='#f0f0f0', gridwidth=1)
fig.update_yaxes(title_text='Log Volume', row=1, col=1)
fig.update_yaxes(title_text='Anomaly Score', row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── Section 3 — Anomaly Log ──
st.subheader("Anomaly Log")

log_display = anomaly_log.copy()
log_display = log_display[log_display.index >= start_date]
log_display = log_display[
    log_display['label'].isin(show_labels)]

if len(log_display) == 0:
    st.info("No anomalies detected in the selected period.")
else:
    log_display_clean = pd.DataFrame({
        'Date'          : log_display.index.strftime(
                              '%d %b %Y'),
        'Day'           : log_display.index.strftime('%A'),
        'Anomaly Score' : log_display['anomaly_score'],
        'STL z-score'   : log_display['z_stl'],
        'Shock z-score' : log_display['z_shock'],
        'Label'         : log_display['label'].str.capitalize(),
    })

    def color_label(val):
        if val == 'Critical':
            return 'color: #dc3545; font-weight: bold'
        elif val == 'Warning':
            return 'color: #856404'
        return ''

    styled = (
        log_display_clean
        .style
        .map(color_label, subset=['Label'])
        .format({
            'Anomaly Score' : '{:.3f}',
            'STL z-score'   : '{:.3f}',
            'Shock z-score' : '{:.3f}',
        })
        .hide(axis='index')
    )

    st.dataframe(styled, use_container_width=True,
                 height=320)
    st.caption(
        f"{len(log_display)} anomalies shown  |  "
        f"Sorted by score (highest first)"
    )

st.divider()


# ── Section 4 — Methodology ──
with st.expander("ℹ️ How this works", expanded=False):
    st.markdown("""
    #### Methodology

    This dashboard monitors HDFC Bank daily trading
    volume on NSE for statistically unusual behavior.

    **What it does:**
    - Decomposes volume into trend, weekly seasonality,
      and residual using STL decomposition
    - Flags days where the residual is unusually large
      relative to the historical distribution
    - Combines two signals: STL residual z-score (60%)
      and day-over-day volume shock z-score (40%)

    **What it does not do:**
    - Predict future prices or volumes
    - Generate buy or sell signals

    **Thresholds:**
    - Warning: score ≥ 2.0 (or ≥ 3.0 on expiry days)
    - Critical: score ≥ 3.0 (or ≥ 4.0 on expiry days)

    *Data source: NSE via Yahoo Finance. Refreshes daily.*
    """)