import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(ticker, start_date, end_date):
    """
    Get historical stock data for a given ticker and date range
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        # Convert MultiIndex to single level
        data.columns = data.columns.get_level_values(0)
    
    # Ensure 'Adj Close' exists
    if 'Adj Close' not in data.columns:
        data['Adj Close'] = data['Close']
        
    return data

def monte_carlo_simulation(ticker_data, num_simulations, time_horizon, last_price=None):
    """
    Run a Monte Carlo simulation for the given stock data
    
    Parameters:
    - ticker_data: DataFrame with historical stock data
    - num_simulations: Number of simulations to run
    - time_horizon: Number of trading days to simulate
    - last_price: Optional last price to use (defaults to last price in ticker_data)
    
    Returns:
    - simulation_df: DataFrame with simulation results
    - metrics: Dict with key metrics from the simulation
    """
    # Calculate daily returns
    returns = ticker_data['Adj Close'].pct_change().dropna()
    
    # Get mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Get last price or use provided last_price
    if last_price is None:
        last_price = ticker_data['Adj Close'].iloc[-1]
    
    # Create empty dataframe for simulation results
    simulation_df = pd.DataFrame()
    
    # Run simulations
    for i in range(num_simulations):
        # Create price series for this simulation
        prices = [last_price]
        
        # Generate price path
        for _ in range(time_horizon):
            # Generate random return from normal distribution
            daily_return = np.random.normal(mu, sigma)
            # Calculate next price
            next_price = prices[-1] * (1 + daily_return)
            prices.append(next_price)
        
        # Add this simulation to the dataframe
        simulation_df[i] = prices
    
    # Calculate metrics
    final_prices = simulation_df.iloc[-1]
    metrics = {
        'mean': final_prices.mean(),
        'median': final_prices.median(),
        'min': final_prices.min(),
        'max': final_prices.max(),
        'std': final_prices.std()
    }
    
    # Calculate percentiles
    for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        metrics[f'percentile_{percentile}'] = final_prices.quantile(percentile/100)
    
    return simulation_df, metrics

def calculate_confidence_interval(simulation_df, confidence_level=95):
    """
    Calculate confidence intervals for the simulation results
    
    Parameters:
    - simulation_df: DataFrame with simulation results
    - confidence_level: Confidence level (default 95%)
    
    Returns:
    - Dictionary with lower and upper bounds
    """
    # Calculate percentiles for confidence interval
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    # Get final values from simulations
    final_values = simulation_df.iloc[-1]
    
    # Calculate bounds
    lower_bound = final_values.quantile(lower_percentile / 100)
    upper_bound = final_values.quantile(upper_percentile / 100)
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'median': final_values.median()
    }

def plot_monte_carlo_results(original_data, simulation_df, confidence_interval, title="Monte Carlo Simulation"):
    """
    Plot Monte Carlo simulation results
    
    Parameters:
    - original_data: DataFrame with original stock data
    - simulation_df: DataFrame with simulation results
    - confidence_interval: Dict with confidence interval values
    - title: Plot title
    
    Returns:
    - matplotlib figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(original_data.index, original_data['Adj Close'], color='blue', label='Historical Data')
    
    # Create future dates for simulation
    last_date = original_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(simulation_df), freq='B')
    
    # Plot simulations (first 100 only for visual clarity)
    for i in range(min(100, simulation_df.shape[1])):
        ax.plot(future_dates, simulation_df[i], color='gray', alpha=0.1)
    
    # Plot median simulation
    ax.plot(future_dates, simulation_df.median(axis=1), color='green', linewidth=2, label='Median Forecast')
    
    # Plot confidence intervals
    lower_series = simulation_df.apply(lambda x: np.percentile(x, (100 - confidence_interval['confidence_level']) / 2), axis=1)
    upper_series = simulation_df.apply(lambda x: np.percentile(x, 100 - (100 - confidence_interval['confidence_level']) / 2), axis=1)
    
    ax.plot(future_dates, lower_series, color='red', linestyle='--', 
            label=f"Lower {confidence_interval['confidence_level']}% CI")
    ax.plot(future_dates, upper_series, color='red', linestyle='--',
            label=f"Upper {confidence_interval['confidence_level']}% CI")
    
    # Fill between confidence intervals
    ax.fill_between(future_dates, lower_series, upper_series, color='red', alpha=0.1)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter('${x:,.2f}')
    
    plt.tight_layout()
    return fig

def analyze_returns(simulation_df, current_price):
    """
    Analyze expected returns from simulation results
    
    Parameters:
    - simulation_df: DataFrame with simulation results
    - current_price: Current stock price
    
    Returns:
    - Dictionary with return metrics
    """
    # Get final prices
    final_prices = simulation_df.iloc[-1]
    
    # Calculate returns
    returns = (final_prices - current_price) / current_price * 100
    
    # Calculate metrics
    metrics = {
        'mean_return': returns.mean(),
        'median_return': returns.median(),
        'min_return': returns.min(),
        'max_return': returns.max(),
        'std_return': returns.std()
    }
    
    # Calculate percentiles
    for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        metrics[f'percentile_{percentile}'] = returns.quantile(percentile/100)
    
    return metrics
