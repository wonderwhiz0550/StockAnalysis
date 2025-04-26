
import yfinance as yf
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration Dictionary for Customization
config = {
    "risk_free_rate": 0.035,
    "market_return": 0.095,
    "terminal_growth_rate": 0.035,
    "default_discount_rate": 0.011,
    "default_analyst_growth_rate": 0.07,
    "high_growth_period": 8,
    "transition_period": 5,
    "sensitivity_range": 0.04,
    "num_monte_carlo_sims": 10000,
    "terminal_method": "exit_multiple",
    "exit_multiple": 20,
    "confidence_level": 0.95,
}

# Helper Functions
def validate_data(data, key, default=None):
    if data is not None and key in data and data[key] is not None:
        return data[key]
    return default if default is not None else np.nan

def fetch_stock_data(ticker, max_retries=3):
    stock = yf.Ticker(ticker)
    retries = 0
    while retries < max_retries:
        try:
            stock_price_data = stock.history(period="1d")
            stock_price = validate_data(stock_price_data, "Close").iloc[-1] if not stock_price_data.empty else np.nan
            income_statement = stock.financials
            cashflow = stock.cashflow
            revenue = income_statement.loc['Total Revenue'].iloc[0]
            operating_cash_flow = cashflow.loc['Operating Cash Flow'].iloc[0]
            capital_expenditure = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
            free_cash_flow = operating_cash_flow - capital_expenditure
            shares_outstanding = stock.info.get("sharesOutstanding", 1)
            debt = validate_data(stock.info, "totalDebt", 0)
            market_cap = validate_data(stock.info, "marketCap", np.nan)
            return stock_price, revenue, free_cash_flow, shares_outstanding, debt, market_cap
        except Exception:
            retries += 1
    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def fetch_analyst_growth_rate(ticker):
    try:
        url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey=demo"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        annual_earnings = data.get("annualEarnings", [])
        if len(annual_earnings) >= 2:
            latest_eps = float(annual_earnings[0]["reportedEPS"])
            previous_eps = float(annual_earnings[1]["reportedEPS"])
            growth_rate = (latest_eps - previous_eps) / previous_eps
            return growth_rate
    except Exception:
        pass
    return config["default_analyst_growth_rate"]

def calculate_discount_rate(ticker, debt, market_cap):
    stock = yf.Ticker(ticker)
    try:
        beta = validate_data(stock.info, "beta", 1)
        cost_of_equity = config["risk_free_rate"] + beta * (config["market_return"] - config["risk_free_rate"])
        cost_of_debt = config["risk_free_rate"]
        if not np.isnan(debt) and not np.isnan(market_cap) and (debt + market_cap) > 0:
            wacc = (cost_of_equity * market_cap + cost_of_debt * debt) / (debt + market_cap)
        else:
            wacc = cost_of_equity
        return wacc
    except Exception:
        return config["default_discount_rate"]

def multi_stage_dcf(revenue, growth_rates, fcf_margin, discount_rate, terminal_growth_rate, shares_outstanding, config):
    high_growth_period = config["high_growth_period"]
    transition_period = config["transition_period"]
    high_growth_rate, transition_growth_rate = growth_rates

    revenues = [revenue * (1 + high_growth_rate)**t for t in range(1, high_growth_period + 1)]
    revenues += [revenues[-1] * (1 + transition_growth_rate)**t for t in range(1, transition_period + 1)]

    if config["terminal_method"] == "perpetual_growth":
        terminal_revenue = revenues[-1] * (1 + terminal_growth_rate)
        terminal_fcf = terminal_revenue * fcf_margin
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    elif config["terminal_method"] == "exit_multiple":
        terminal_ebitda = revenues[-1] * fcf_margin
        terminal_value = terminal_ebitda * config["exit_multiple"]
    else:
        raise ValueError("Invalid terminal method.")

    fcfs = [rev * fcf_margin for rev in revenues]
    npv = sum(fcf / (1 + discount_rate)**t for t, fcf in enumerate(fcfs, start=1))
    npv += terminal_value / (1 + discount_rate)**(high_growth_period + transition_period)
    return npv / shares_outstanding

def monte_carlo_simulation(stock_price, revenue, fcf_margin, discount_rate, terminal_growth_rate, shares_outstanding, config):
    simulated_prices = []
    for _ in range(config["num_monte_carlo_sims"]):
        growth_rate = np.clip(np.random.normal(0.05, config["sensitivity_range"]), -0.2, 0.5)
        discount_rate_sim = np.clip(np.random.normal(discount_rate, config["sensitivity_range"] * discount_rate), 0.01, 0.2)
        terminal_growth_rate_sim = np.clip(np.random.normal(terminal_growth_rate, config["sensitivity_range"] * terminal_growth_rate), 0.01, 0.06)

        try:
            price = multi_stage_dcf(
                revenue, [growth_rate, growth_rate * 0.5], fcf_margin, discount_rate_sim,
                terminal_growth_rate_sim, shares_outstanding, config
            )
            if price > 0:
                simulated_prices.append(price)
        except ValueError:
            continue
    return simulated_prices

def compare_prices(stock_price, implied_price):
    if stock_price < implied_price * 0.9:
        return "Undervalued"
    elif stock_price > implied_price * 1.1:
        return "Overvalued"
    return "Fairly Priced"

def evaluate_stock(ticker, config):
    stock_price, revenue, free_cash_flow, shares_outstanding, debt, market_cap = fetch_stock_data(ticker)
    if any(np.isnan(x) for x in [stock_price, revenue, free_cash_flow, shares_outstanding]):
        return None, "Failed to fetch stock data", None

    if revenue == 0:
        return None, "Revenue is zero", None

    fcf_margin = free_cash_flow / revenue
    discount_rate = calculate_discount_rate(ticker, debt, market_cap)
    terminal_growth_rate = config["terminal_growth_rate"]

    simulated_prices = monte_carlo_simulation(
        stock_price, revenue, fcf_margin, discount_rate, terminal_growth_rate, shares_outstanding, config
    )

    if not simulated_prices:
        return None, "No valid simulations", None

    mean_price = np.mean(simulated_prices)
    valuation_status = compare_prices(stock_price, mean_price)

    plt.figure(figsize=(8, 5))
    plt.hist(simulated_prices, bins=50, alpha=0.75, color='white', edgecolor='black')
    plt.axvline(stock_price, color='red', linestyle='dashed', label="Stock Price")
    plt.axvline(mean_price, color='green', linestyle='dashed', label="Mean Simulated Price")
    lower_bound = np.percentile(simulated_prices, (1 - config["confidence_level"]) / 2 * 100)
    upper_bound = np.percentile(simulated_prices, (1 + config["confidence_level"]) / 2 * 100)
    plt.axvline(lower_bound, color='blue', linestyle='dotted', label="CI Lower")
    plt.axvline(upper_bound, color='blue', linestyle='dotted', label="CI Upper")
    plt.title(f"Monte Carlo Valuation for {ticker}")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("valuation_plot.png")
    plt.close()

    output = {
        "stock_price": stock_price,
        "mean_simulated_price": mean_price,
        "valuation_status": valuation_status,
        "revenue": revenue,
        "free_cash_flow": free_cash_flow,
        "fcf_margin": fcf_margin
    }

    return output, "Success", "valuation_plot.png"

# New: Technical Analysis Functions
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def perform_technical_analysis(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

    ax0.plot(df.index, df['Close'], label='Close', color='black')
    ax0.plot(df.index, df['SMA20'], label='SMA20', color='blue')
    ax0.plot(df.index, df['SMA50'], label='SMA50', color='red')
    ax0.set_title(f'{ticker} Price with SMA20 & SMA50')
    ax0.legend()
    ax0.grid(True)

    ax1.plot(df.index, df['RSI'], color='purple', label='RSI')
    ax1.axhline(70, color='red', linestyle='--')
    ax1.axhline(30, color='green', linestyle='--')
    ax1.set_title('RSI')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange')
    ax2.bar(df.index, df['MACD'] - df['Signal_Line'], label='MACD Histogram', color='gray')
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = "technical_analysis_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

__all__ = ["evaluate_stock", "config", "perform_technical_analysis"]
