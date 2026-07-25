# Statistical Time Series Forecasting & Risk Analytics Engine

ForecastIQ is an end-to-end financial analytics platform that forecasts market trends and quantifies portfolio risk using classical statistical time series methods, paired with an LLM-powered executive reporting layer.

---

## Overview

ForecastIQ processes historical OHLCV (Open-High-Low-Close-Volume) data, generates statistically validated price forecasts, and computes institutional-grade risk metrics - all surfaced through an interactive Streamlit dashboard with AI-generated executive summaries.

## Features

- **Data Processing**: Ingests and cleans 12.5K+ OHLCV records, engineering 5 financial features for downstream modeling
- **Stationarity Testing**: Runs ADF (Augmented Dickey-Fuller) and KPSS tests to determine series stationarity and the ARIMA differencing parameter (d=1)
- **Forecasting Engine**: Optimizes ARIMA(1,1,1) via AIC comparison across 6 candidate models, achieving 0.63%-1.13% MAPE
- **30-Day Forecasts**: Generates price forecasts with 95% confidence intervals
- **Risk Engine**: Computes 6 institutional-grade risk metrics -
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Sharpe Ratio
  - Maximum Drawdown
  - Rolling Volatility
  - Annual Return
- **AI-Generated Reports**: Integrates Groq Llama-3.3-70B to auto-generate executive summaries from 17 pre-computed metrics
- **Interactive Dashboard**: 4-page Streamlit application for exploring forecasts, risk metrics, and AI-generated insights

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Time Series Modeling | Statsmodels (ARIMA), ADF/KPSS tests |
| LLM Integration | Groq API (Llama-3.3-70B) |
| Data Processing | Pandas, NumPy |
| Visualization | Streamlit, Matplotlib/Plotly |

## Methodology

1. **Data Ingestion** - Load and clean OHLCV records
2. **Feature Engineering** - Derive 5 financial features from raw price/volume data
3. **Stationarity Testing** - ADF & KPSS tests confirm the series requires first-order differencing (d=1)
4. **Model Selection** - Compare 6 ARIMA model configurations via AIC; select ARIMA(1,1,1) as optimal
5. **Forecasting** - Generate 30-day forward price forecasts with 95% confidence intervals
6. **Risk Quantification** - Compute VaR, CVaR, Sharpe Ratio, Max Drawdown, Rolling Volatility, and Annual Return
7. **Reporting** - Feed pre-computed metrics into Groq Llama-3.3-70B to generate plain-language executive summaries
8. **Visualization** - Present all outputs through a 4-page Streamlit dashboard

## Results

- Achieved **0.63%-1.13% MAPE** across forecast horizons
- Delivered 30-day forecasts with **95% confidence intervals**
- Consolidated **17 pre-computed metrics** into automated executive reports

## Project Structure

```
forecastiq/
├── data/                  # Raw and processed OHLCV data
├── notebooks/              # EDA and model development notebooks
├── src/
│   ├── preprocessing.py    # Data cleaning & feature engineering
│   ├── stationarity.py     # ADF & KPSS tests
│   ├── forecasting.py      # ARIMA model selection & forecasting
│   ├── risk_engine.py      # VaR, CVaR, Sharpe, Drawdown, Volatility, Returns
│   └── report_generator.py # Groq Llama-3.3-70B integration
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```


## Author

**Shah Aanchal Sachinkumar**
B.Tech (Hons.) Aerospace Engineering, IIT Kharagpur
