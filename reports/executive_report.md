# ForecastIQ — Executive Report

*Generated: June 24, 2026 | Model: Groq llama-3.3-70b-versatile*

---

## Executive Snapshot
- American Express (AXP) has the **13.12%** highest annualised return among the given assets, but not the highest overall.
- Apple (AAPL) has the **28.47%** highest volatility.
- Microsoft (MSFT) has the **0.724** highest Sharpe Ratio.
- American Express (AXP) has the **49.64%** largest maximum drawdown.
- S&P 500 (GSPC) has the **0.63%** lowest ARIMA MAPE.

## 1. Market Performance Summary
| Asset | Annual Return | Volatility | Sharpe Ratio |
|-------|---------------:|-----------:|-------------:|
| AXP   | 13.12%         | 30.68%     | 0.297        |
| AAPL  | 23.43%         | 28.47%     | 0.683        |
| MSFT  | 23.65%         | 27.13%     | 0.724        |
| GSPC  | 10.57%         | 17.89%     | 0.367        |
| IXIC  | 14.2%          | 21.41%     | 0.476        |
- AXP has an annualised return of **13.12%**.
- MSFT has the highest annualised return among the given assets at **23.65%**.
- The S&P 500 (GSPC) has the lowest volatility at **17.89%**.

## 2. Risk Assessment
- 95% VaR for AXP is **2.78%**.
- 95% CVaR for AXP is **4.48%**.
- Maximum Drawdown for AXP is **49.64%**.
### Key Risks
- AXP has a **49.64%** maximum drawdown.
- MSFT has a **2.69%** 95% VaR.
- S&P 500 (GSPC) has a **1.71%** 95% VaR.

## 3. Forecasting Model Evaluation
| Asset | ARIMA MAPE | Beats Naive? |
|-------|------------|--------------|
| AXP   | 1.13%      | True         |
| AAPL  | 1.02%      | False        |
| MSFT  | 1.08%      | False        |
| GSPC  | 0.63%      | False        |
| IXIC  | 0.87%      | False        |
- The training period for ARIMA walk-forward validation was **2015-2022**.
- The testing period was **2023-2024**.
- Only AXP marginally beat the naive baseline with an ARIMA MAPE of **1.13%**.
- S&P 500 (GSPC) has the lowest ARIMA MAPE at **0.63%**.
- MSFT has an ARIMA MAPE of **1.08%**.

## 4. Investment Conclusions & Recommendations
### Conservative Investors
- Consider S&P 500 (GSPC) for its **10.57%** annualised return and **17.89%** volatility.
- AXP has a **13.12%** annualised return, but **30.68%** volatility.
### Growth Investors
- Apple (AAPL) has a **23.43%** annualised return and **28.47%** volatility.
- Microsoft (MSFT) has a **23.65%** annualised return and **27.13%** volatility.
### Higher-Risk Investors
- NASDAQ (IXIC) has a **14.2%** annualised return and **21.41%** volatility.
- AXP has a **13.12%** annualised return, but **49.64%** maximum drawdown.
### Overall Recommendation
- Consider MSFT for its **0.724** Sharpe Ratio and **23.65%** annualised return.
- S&P 500 (GSPC) has a **0.367** Sharpe Ratio and **10.57%** annualised return.

## Disclaimer
This report is generated from historical market data, forecasting models, and risk analytics and does not constitute investment advice. The information provided is based on the computed metrics and additional context, and investors should consult with a financial advisor before making any investment decisions.

---
*All quantitative metrics were computed independently using statistical models (ARIMA, historical simulation VaR, Sharpe ratio). This report is generated for educational purposes only and does not constitute financial advice.*
