# ============================================================
# ForecastIQ | Day 8: Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ForecastIQ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }
    .metric-card.green  { border-left-color: #2ca02c; }
    .metric-card.red    { border-left-color: #d62728; }
    .metric-card.orange { border-left-color: #ff7f0e; }
    .metric-label { font-size: 12px; color: #666; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #111; }
    .metric-sub   { font-size: 11px; color: #999; margin-top: 2px; }
    .section-header {
        font-size: 13px; font-weight: 600;
        color: #555; letter-spacing: 0.08em;
        text-transform: uppercase; margin: 20px 0 10px;
    }
    div[data-testid="stSidebar"] { background: #0f1117; }
    div[data-testid="stSidebar"] * { color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
TICKERS = {
    "AXP":  "American Express",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GSPC": "S&P 500",
    "IXIC": "NASDAQ",
}

COLORS = {
    "AXP":  "#1f77b4",
    "AAPL": "#ff7f0e",
    "MSFT": "#2ca02c",
    "GSPC": "#9467bd",
    "IXIC": "#d62728",
}

RISK_METRICS = {
    "AXP":  {"ann_return": 13.12, "ann_vol": 30.68, "sharpe": 0.297,
              "var_95": 2.78, "cvar_95": 4.48, "max_drawdown": 49.64},
    "AAPL": {"ann_return": 23.43, "ann_vol": 28.47, "sharpe": 0.683,
              "var_95": 2.74, "cvar_95": 4.17, "max_drawdown": 38.52},
    "MSFT": {"ann_return": 23.65, "ann_vol": 27.13, "sharpe": 0.724,
              "var_95": 2.69, "cvar_95": 3.97, "max_drawdown": 37.15},
    "GSPC": {"ann_return": 10.57, "ann_vol": 17.89, "sharpe": 0.367,
              "var_95": 1.71, "cvar_95": 2.79, "max_drawdown": 33.92},
    "IXIC": {"ann_return": 14.20, "ann_vol": 21.41, "sharpe": 0.476,
              "var_95": 2.21, "cvar_95": 3.33, "max_drawdown": 36.40},
}

PROCESSED_DIR = "data/processed"
ARIMA_ORDER   = (1, 1, 1)

# ── Data loader (cached) ─────────────────────────────────────
@st.cache_data
def load_data():
    data = {}
    for ticker in TICKERS:
        path = f"{PROCESSED_DIR}/{ticker}_processed.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            df["Log_Return"]   = np.log(df["Close"] / df["Close"].shift(1))
            df["Simple_Return"] = df["Close"].pct_change()
            df["Vol_30d"] = df["Log_Return"].rolling(30).std() * np.sqrt(252) * 100
            df["Vol_90d"] = df["Log_Return"].rolling(90).std() * np.sqrt(252) * 100
            data[ticker] = df
    return data

@st.cache_data
def run_forecast(ticker, steps=30):
    data = load_data()
    if ticker not in data:
        return None, None
    series = data[ticker]["Close"].dropna()
    model  = ARIMA(series, order=ARIMA_ORDER).fit()
    fc     = model.get_forecast(steps=steps)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)
    future  = pd.bdate_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=steps
    )
    forecast_df = pd.DataFrame({
        "Forecast": fc_mean.values,
        "Lower_95": fc_ci.iloc[:, 0].values,
        "Upper_95": fc_ci.iloc[:, 1].values,
    }, index=future)
    return series, forecast_df

def drawdown_series(prices):
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max * 100

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ForecastIQ")
    st.markdown("*Financial Forecasting & Risk Analytics*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Market Overview",
         "Forecast Dashboard",
         "Risk Dashboard",
         "AI Executive Report"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Select Asset**")
    selected_ticker = st.selectbox(
        "Asset", list(TICKERS.keys()),
        format_func=lambda x: f"{x} — {TICKERS[x]}",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Analysis Period**")
    period = st.select_slider(
        "Period",
        options=["1Y", "2Y", "5Y", "Full"],
        value="Full",
        label_visibility="collapsed"
    )

    period_days = {"1Y": 252, "2Y": 504, "5Y": 1260, "Full": 9999}

    st.markdown("---")
    st.caption("Analysis period: Jan 2015 – Dec 2024")
    st.caption("Model: ARIMA(1,1,1)")
    st.caption("Data source: Yahoo Finance")

# ── Load data ────────────────────────────────────────────────
data = load_data()

if not data:
    st.error("No processed data found. Run notebooks 1–3 first.")
    st.stop()

df      = data[selected_ticker]
n_days  = period_days[period]
df_view = df.tail(min(n_days, len(df)))

# ════════════════════════════════════════════════════════════
# PAGE 1: MARKET OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Market Overview":

    st.title("Market Overview")
    st.markdown(f"Showing **{TICKERS[selected_ticker]}** · Period: **{period}**")

    # ── KPI cards ─────────────────────────────────────────
    latest_close  = df["Close"].iloc[-1]
    prev_close    = df["Close"].iloc[-2]
    daily_return  = (latest_close - prev_close) / prev_close * 100
    vol_30d       = df["Vol_30d"].iloc[-1]
    risk_level    = "High" if vol_30d > 30 else ("Medium" if vol_30d > 20 else "Low")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Latest Close", f"${latest_close:.2f}",
                  delta=f"{daily_return:+.2f}% today")
    with c2:
        st.metric("Daily Return", f"{daily_return:+.2f}%",
                  delta="vs previous close")
    with c3:
        st.metric("30d Volatility", f"{vol_30d:.1f}%",
                  delta=risk_level, delta_color="off")
    with c4:
        sharpe = RISK_METRICS[selected_ticker]["sharpe"]
        st.metric("Sharpe Ratio", f"{sharpe:.3f}",
                  delta="10-year period")

    st.markdown("---")

    # ── Price trend ───────────────────────────────────────
    st.markdown('<p class="section-header">Price Trend</p>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_view.index, y=df_view["Close"],
        name="Close Price",
        line=dict(color=COLORS[selected_ticker], width=2),
        fill="tozeroy",
        fillcolor=COLORS[selected_ticker].replace(
            "#", "rgba(").replace("b4", "b4,0.08)") + ""
        if False else f"rgba(31,119,180,0.08)"
    ))
    fig.update_layout(
        template="plotly_white", height=320,
        margin=dict(t=20, b=20),
        xaxis_title="", yaxis_title="Price (USD)",
        hovermode="x unified", showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Return distribution + Normalised comparison ───────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">Return Distribution</p>',
                    unsafe_allow_html=True)
        returns = df_view["Log_Return"].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=returns * 100, nbinsx=60,
            marker_color=COLORS[selected_ticker],
            opacity=0.8, name="Log Returns"
        ))
        fig2.add_vline(x=0, line_dash="dash", line_color="gray")
        fig2.update_layout(
            template="plotly_white", height=280,
            margin=dict(t=20, b=20),
            xaxis_title="Daily Log Return (%)",
            yaxis_title="Frequency", showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Stats below chart
        s1, s2, s3 = st.columns(3)
        s1.metric("Skewness", f"{returns.skew():.3f}")
        s2.metric("Kurtosis", f"{returns.kurt():.3f}")
        s3.metric("Ann. Return", f"{RISK_METRICS[selected_ticker]['ann_return']:.1f}%")

    with col_right:
        st.markdown('<p class="section-header">Normalised Performance (Base=100)</p>',
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        for t in TICKERS:
            if t in data:
                d     = data[t].tail(min(n_days, len(data[t])))
                norm  = (d["Close"] / d["Close"].iloc[0]) * 100
                width = 2.5 if t == selected_ticker else 1
                fig3.add_trace(go.Scatter(
                    x=norm.index, y=norm.values,
                    name=TICKERS[t],
                    line=dict(color=COLORS[t], width=width)
                ))
        fig3.update_layout(
            template="plotly_white", height=280,
            margin=dict(t=20, b=20),
            xaxis_title="", yaxis_title="Indexed Price",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25, font=dict(size=10))
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Rolling volatility ────────────────────────────────
    st.markdown('<p class="section-header">Rolling Volatility</p>',
                unsafe_allow_html=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_view.index, y=df_view["Vol_90d"],
        name="90-day", line=dict(color="#2ca02c", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=df_view.index, y=df_view["Vol_30d"],
        name="30-day", line=dict(color="#ff7f0e", width=1.5)
    ))
    fig4.update_layout(
        template="plotly_white", height=250,
        margin=dict(t=20, b=20),
        xaxis_title="", yaxis_title="Ann. Volatility (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.3, font=dict(size=10))
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2: FORECAST DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "Forecast Dashboard":

    st.title("Forecast Dashboard")
    st.markdown(f"ARIMA(1,1,1) · **{TICKERS[selected_ticker]}** · 30-day horizon")

    with st.spinner("Fitting ARIMA model..."):
        close_series, forecast_df = run_forecast(selected_ticker, steps=30)

    if forecast_df is None:
        st.error("Could not generate forecast.")
        st.stop()

    # ── Forecast KPIs ─────────────────────────────────────
    last_price  = close_series.iloc[-1]
    fc_day1     = forecast_df["Forecast"].iloc[0]
    fc_day30    = forecast_df["Forecast"].iloc[-1]
    ci_low      = forecast_df["Lower_95"].iloc[-1]
    ci_high     = forecast_df["Upper_95"].iloc[-1]
    ci_width    = ci_high - ci_low
    pct_change  = (fc_day30 - last_price) / last_price * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close",       f"${last_price:.2f}")
    c2.metric("Day 1 Forecast",   f"${fc_day1:.2f}",
              delta=f"{(fc_day1-last_price)/last_price*100:+.2f}%")
    c3.metric("Day 30 Forecast",  f"${fc_day30:.2f}",
              delta=f"{pct_change:+.2f}%")
    c4.metric("95% CI Width (D30)", f"${ci_width:.2f}",
              delta=f"${ci_low:.0f} – ${ci_high:.0f}", delta_color="off")

    st.markdown("---")

    # ── Forecast chart ────────────────────────────────────
    st.markdown('<p class="section-header">Price Forecast with 95% Confidence Interval</p>',
                unsafe_allow_html=True)

    history = close_series[close_series.index >= close_series.index[-1] - pd.DateOffset(days=180)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        name="Historical", line=dict(color=COLORS[selected_ticker], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["Upper_95"]) + list(forecast_df["Lower_95"][::-1]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval"
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["Forecast"],
        name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")
    ))
    fig.add_vline(
        x=str(close_series.index[-1].date()),
        line_dash="dot", line_color="gray",
        annotation_text="Forecast start"
    )
    fig.update_layout(
        template="plotly_white", height=380,
        margin=dict(t=20, b=20),
        xaxis_title="", yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, font=dict(size=10))
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table + model note ───────────────────────
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown('<p class="section-header">Forecast Table</p>',
                    unsafe_allow_html=True)
        show_rows = forecast_df.iloc[[0, 4, 9, 19, 29]].copy()
        show_rows.index = show_rows.index.strftime("%b %d, %Y")
        show_rows.columns = ["Forecast $", "Lower $", "Upper $"]
        show_rows = show_rows.round(2)
        st.dataframe(show_rows, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">Model Validation Summary</p>',
                    unsafe_allow_html=True)
        val_data = {
            "Metric":      ["RMSE", "MAE", "MAPE", "Beats Naive?"],
            "Naive":       ["varies", "varies", "varies", "Baseline"],
            "ARIMA (WFV)": [
                "~3.19" if selected_ticker == "AXP" else "~2.55",
                "—", 
                f"{RISK_METRICS[selected_ticker].get('var_95', 1.0):.2f}%",
                "Yes" if selected_ticker == "AXP" else "Marginal"
            ]
        }
        mape_map   = {"AXP": 1.13, "AAPL": 1.02, "MSFT": 1.08,
                      "GSPC": 0.63, "IXIC": 0.87}
        rmse_naive = {"AXP": 3.19, "AAPL": 2.53, "MSFT": 4.96,
                      "GSPC": 39.45, "IXIC": 171.07}
        rmse_arima = {"AXP": 3.19, "AAPL": 2.55, "MSFT": 4.96,
                      "GSPC": 39.81, "IXIC": 172.00}

        val_df = pd.DataFrame({
            "Metric":      ["RMSE (Naive)", "RMSE (ARIMA)", "MAPE (ARIMA)", "Beats Naive?"],
            "Value": [
                f"{rmse_naive[selected_ticker]:.2f}",
                f"{rmse_arima[selected_ticker]:.2f}",
                f"{mape_map[selected_ticker]:.2f}%",
                "Yes" if selected_ticker == "AXP" else "No (marginal)"
            ]
        })
        st.dataframe(val_df, use_container_width=True, hide_index=True)

        st.info(
            "ARIMA on stock prices approaches the random walk baseline — "
            "consistent with the Efficient Market Hypothesis. "
            "Confidence intervals are the primary output of value."
        )

# ════════════════════════════════════════════════════════════
# PAGE 3: RISK DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "Risk Dashboard":

    st.title("Risk Dashboard")
    st.markdown(f"**{TICKERS[selected_ticker]}** · Full period 2015–2024")

    rm = RISK_METRICS[selected_ticker]

    # ── Risk KPI cards ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised Volatility", f"{rm['ann_vol']:.2f}%")
    c2.metric("95% VaR (1-day)",       f"{rm['var_95']:.2f}%",
              delta="loss threshold", delta_color="off")
    c3.metric("Max Drawdown",          f"{rm['max_drawdown']:.2f}%",
              delta="peak-to-trough", delta_color="off")
    c4.metric("Sharpe Ratio",          f"{rm['sharpe']:.3f}",
              delta="vs 4% risk-free")

    st.markdown("---")

    # ── Cross-asset risk table ────────────────────────────
    st.markdown('<p class="section-header">Cross-Asset Risk Comparison</p>',
                unsafe_allow_html=True)

    risk_table = pd.DataFrame(RISK_METRICS).T.reset_index()
    risk_table.columns = ["Ticker", "Ann.Return%", "Ann.Vol%",
                           "Sharpe", "VaR95%", "CVaR95%", "MaxDD%"]
    risk_table["Ticker"] = risk_table["Ticker"].map(
        lambda x: f"{x} — {TICKERS[x]}"
    )
    risk_table = risk_table.round(2)
    st.dataframe(
        risk_table.style.highlight_max(
            subset=["Ann.Return%", "Sharpe"],
            color="#d4edda"
        ).highlight_min(
            subset=["Ann.Vol%", "VaR95%", "MaxDD%"],
            color="#d4edda"
        ).highlight_max(
            subset=["Ann.Vol%", "MaxDD%"],
            color="#f8d7da"
        ),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ── Drawdown chart ────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-header">Drawdown Curve</p>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        for t in TICKERS:
            if t in data:
                dd    = drawdown_series(data[t]["Close"])
                width = 2.5 if t == selected_ticker else 1
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values,
                    name=TICKERS[t],
                    line=dict(color=COLORS[t], width=width)
                ))
        fig.update_layout(
            template="plotly_white", height=300,
            margin=dict(t=10, b=10),
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.3, font=dict(size=9))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">VaR vs CVaR Comparison</p>',
                    unsafe_allow_html=True)
        ticker_names = [TICKERS[t] for t in TICKERS]
        var_vals     = [RISK_METRICS[t]["var_95"]  for t in TICKERS]
        cvar_vals    = [RISK_METRICS[t]["cvar_95"] for t in TICKERS]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="95% VaR", x=ticker_names, y=var_vals,
            marker_color="#1f77b4", opacity=0.85
        ))
        fig2.add_trace(go.Bar(
            name="95% CVaR", x=ticker_names, y=cvar_vals,
            marker_color="#d62728", opacity=0.85
        ))
        fig2.update_layout(
            template="plotly_white", height=300,
            margin=dict(t=10, b=10),
            yaxis_title="Loss Threshold (%)",
            barmode="group",
            legend=dict(orientation="h", y=-0.3, font=dict(size=9))
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Rolling volatility timeline ───────────────────────
    st.markdown('<p class="section-header">Rolling Volatility Timeline — All Assets</p>',
                unsafe_allow_html=True)

    fig3 = go.Figure()
    for t in TICKERS:
        if t in data:
            width = 2.5 if t == selected_ticker else 1
            fig3.add_trace(go.Scatter(
                x=data[t].index,
                y=data[t]["Vol_30d"],
                name=TICKERS[t],
                line=dict(color=COLORS[t], width=width)
            ))

    fig3.add_vline(x="2020-03-16", line_dash="dash",
                   line_color="gray", annotation_text="COVID")
    fig3.add_vline(x="2022-06-16", line_dash="dash",
                   line_color="gray", annotation_text="2022 Bear")
    fig3.update_layout(
        template="plotly_white", height=280,
        margin=dict(t=10, b=10),
        yaxis_title="Ann. Volatility (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25, font=dict(size=9))
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4: AI EXECUTIVE REPORT
# ════════════════════════════════════════════════════════════
elif page == "AI Executive Report":

    st.title("AI Executive Report")
    st.markdown("*Powered by Groq · llama-3.3-70b-versatile*")
    st.markdown("---")

    # Check for saved report first
    report_path = "reports/executive_report.md"

    col_l, col_r = st.columns([3, 1])

    with col_r:
        st.markdown("**Options**")
        use_saved   = st.checkbox("Use saved report", value=True)
        show_assets = st.checkbox("Show asset summaries", value=True)
        groq_key    = st.text_input(
            "Groq API Key (for regeneration)",
            type="password", placeholder="gsk_..."
        )
        regenerate = st.button("Regenerate Report", use_container_width=True)

    with col_l:
        if use_saved and os.path.exists(report_path) and not regenerate:
            with open(report_path, "r", encoding="utf-8") as f:
                saved_report = f.read()
            st.markdown(saved_report)

        elif regenerate or not os.path.exists(report_path):
            api_key = groq_key or os.getenv("GROQ_API_KEY", "")
            if not api_key or api_key == "your-api-key-here":
                st.warning("Enter your Groq API key in the sidebar to generate a report.")
                st.stop()

            def build_metrics_block():
                lines = []
                for ticker, m in RISK_METRICS.items():
                    lines.append(f"""
  {TICKERS[ticker]} ({ticker}):
    - Annualised Return:     {m['ann_return']}%
    - Annualised Volatility: {m['ann_vol']}%
    - Sharpe Ratio:          {m['sharpe']}
    - 95% VaR (1-day):       {m['var_95']}%
    - 95% CVaR:              {m['cvar_95']}%
    - Max Drawdown:          {m['max_drawdown']}%""")
                return "\n".join(lines)

            PROMPT = f"""
You are a senior quantitative analyst. Write a professional executive report.

STRICT RULES:
1. Use ONLY the numbers provided below.
2. Use exactly these four sections:
   ## 1. Market Performance Summary
   ## 2. Risk Assessment
   ## 3. Forecasting Model Evaluation
   ## 4. Investment Conclusions & Recommendations
3. 2-3 paragraphs per section.
4. End with a one-paragraph disclaimer.

COMPUTED METRICS:
{build_metrics_block()}

CONTEXT:
- ARIMA walk-forward MAPE: AXP 1.13%, AAPL 1.02%, MSFT 1.08%, GSPC 0.63%, IXIC 0.87%
- Only AXP marginally beat the naive random walk baseline
- Period includes COVID-19 crash (March 2020) and 2022 bear market
- All Sharpe ratios below 1.0 — realistic for this volatile decade

Write the report now:
"""
            with st.spinner("Generating report with Groq..."):
                client   = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system",
                         "content": "You are a senior quantitative analyst. "
                                    "Use only provided numbers. Never invent figures."},
                        {"role": "user", "content": PROMPT}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
                report_text = response.choices[0].message.content

            st.markdown(report_text)

            os.makedirs("reports", exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            st.success("Report saved to reports/executive_report.md")

        else:
            st.info("Check 'Use saved report' or click 'Regenerate Report'.")

    # ── Asset summaries ───────────────────────────────────
    if show_assets:
        st.markdown("---")
        st.markdown("### Per-Asset Summaries")

        summaries_path = "reports/asset_summaries.md"
        if os.path.exists(summaries_path):
            with open(summaries_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.caption("Run notebook 07_llm_report.ipynb to generate asset summaries.")