import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set the app title and description
st.title('Stock Analysis Tool')
st.write('This app allows you to analyze stocks and run Monte Carlo simulations.')

# Sidebar inputs
with st.sidebar:
    st.header('Input Parameters')
    
    # Stock symbol input
    ticker = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG)', 'AAPL')
    
    # Time period selection
    period_options = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    period = st.selectbox('Select Time Period', period_options, index=3)
    
    # Date range selection (only visible if 'Custom' is selected)
    use_custom_dates = st.checkbox('Use Custom Date Range')
    
    if use_custom_dates:
        today = date.today()
        start_date = st.date_input('Start Date', today - timedelta(days=365))
        end_date = st.date_input('End Date', today)
    
    # Monte Carlo simulation parameters
    st.header('Monte Carlo Simulation')
    run_simulation = st.checkbox('Run Monte Carlo Simulation', value=True)
    
    if run_simulation:
        num_simulations = st.slider('Number of Simulations', min_value=10, max_value=1000, value=200)
        num_days = st.slider('Number of Days to Predict', min_value=30, max_value=365, value=252)
        confidence_level = st.slider('Confidence Level (%)', min_value=80, max_value=99, value=95)

# Function to load stock data
@st.cache_data(ttl=3600)
def load_data(ticker, period=None, start=None, end=None):
    if start and end:
        data = yf.download(ticker, start=start, end=end)
    else:
        data = yf.download(ticker, period=period)
    
    return data

# Function to calculate daily returns
def calculate_returns(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    return data

# Function to run Monte Carlo simulation
def run_monte_carlo(data, num_simulations, num_days, confidence_level):
    # Get the last closing price
    last_price = data['Adj Close'].iloc[-1]
    
    # Calculate daily returns
    daily_returns = data['Daily Return'].dropna()
    
    # Calculate mean and standard deviation of daily returns
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    
    # Create simulation results dataframe
    simulation_df = pd.DataFrame()
    
    # Run simulations
    for i in range(num_simulations):
        # Create list to store price series
        price_series = [last_price]
        
        # Generate future prices
        for _ in range(num_days):
            # Generate daily returns using random normal distribution
            daily_return = np.random.normal(mu, sigma)
            # Calculate next price
            price_series.append(price_series[-1] * (1 + daily_return))
        
        # Store simulation results
        simulation_df[i] = price_series
    
    # Calculate confidence intervals
    lower_bound = (100 - confidence_level) / 2
    upper_bound = 100 - lower_bound
    
    # Get percentiles
    lower_percentile = simulation_df.iloc[-1].quantile(lower_bound / 100)
    upper_percentile = simulation_df.iloc[-1].quantile(upper_bound / 100)
    median = simulation_df.iloc[-1].median()
    
    return simulation_df, lower_percentile, upper_percentile, median

# Load data based on inputs
if use_custom_dates:
    data = load_data(ticker, start=start_date, end=end_date)
else:
    data = load_data(ticker, period=period)

# If data is available, display analysis
if not data.empty:
    # Display basic stock information
    st.header(f'{ticker} Stock Analysis')
    
    # Calculate returns
    data = calculate_returns(data)
    
    # Display stock price chart
    st.subheader('Stock Price History')
    fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Adjusted Close Price')
    st.plotly_chart(fig)
    
    # Display basic statistics
    st.subheader('Basic Statistics')
    current_price = data['Adj Close'].iloc[-1]
    price_change = data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]
    percent_change = (price_change / data['Adj Close'].iloc[0]) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Price Change", f"${price_change:.2f}")
    col3.metric("Percent Change", f"{percent_change:.2f}%")
    
    # Run Monte Carlo simulation if requested
    if run_simulation:
        st.header('Monte Carlo Simulation')
        st.write(f'Running {num_simulations} simulations for {num_days} trading days with {confidence_level}% confidence level')
        
        # Run simulation
        simulation_df, lower_bound, upper_bound, median = run_monte_carlo(data, num_simulations, num_days, confidence_level)
        
        # Create figure for simulation results
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add historical price
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['Adj Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            )
        )
        
        # Add Monte Carlo simulations (first 100 only for visual clarity)
        for i in range(min(100, num_simulations)):
            fig.add_trace(
                go.Scatter(
                    x=pd.date_range(start=data.index[-1], periods=num_days+1, freq='B'),
                    y=simulation_df[i],
                    mode='lines',
                    opacity=0.1,
                    line=dict(color='green'),
                    name=f'Simulation {i}',
                    showlegend=False
                )
            )
        
        # Calculate and display confidence intervals
        future_dates = pd.date_range(start=data.index[-1], periods=num_days+1, freq='B')
        
        # Add median line
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=simulation_df.median(axis=1),
                mode='lines',
                name='Median Forecast',
                line=dict(color='green', width=2)
            )
        )
        
        # Add confidence interval
        lower_bound_pct = (100 - confidence_level) / 2
        upper_bound_pct = 100 - lower_bound_pct
        
        lower_bound_series = simulation_df.apply(lambda x: np.percentile(x, lower_bound_pct), axis=1)
        upper_bound_series = simulation_df.apply(lambda x: np.percentile(x, upper_bound_pct), axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_bound_series,
                mode='lines',
                name=f'Upper {confidence_level}% CI',
                line=dict(color='rgba(0,100,0,0.3)', width=1, dash='dash')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_bound_series,
                mode='lines',
                name=f'Lower {confidence_level}% CI',
                line=dict(color='rgba(0,100,0,0.3)', width=1, dash='dash'),
                fill='tonexty', 
                fillcolor='rgba(0,100,0,0.1)'
            )
        )
        
        fig.update_layout(
            title=f'Monte Carlo Simulation: {ticker} Stock Price Projection',
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            hovermode='x',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig)
        
        # Display statistical summary
        st.subheader('Statistical Summary of Simulations')
        final_price = simulation_df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Price", f"${median:.2f}")
        col2.metric(f"Lower {confidence_level}% Bound", f"${lower_bound:.2f}")
        col3.metric(f"Upper {confidence_level}% Bound", f"${upper_bound:.2f}")
        
        # Calculate returns at different percentiles
        expected_return = (median - current_price) / current_price * 100
        lower_return = (lower_bound - current_price) / current_price * 100
        upper_return = (upper_bound - current_price) / current_price * 100
        
        st.write(f"Expected return: {expected_return:.2f}%")
        st.write(f"Return range ({confidence_level}% confidence): {lower_return:.2f}% to {upper_return:.2f}%")
        
        # Histogram of final prices
        st.subheader('Distribution of Final Simulated Prices')
        fig = px.histogram(final_price, nbins=50, title=f'Histogram of {ticker} Simulated Prices after {num_days} Days')
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", annotation_text=f"Lower {confidence_level}% CI")
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", annotation_text=f"Upper {confidence_level}% CI")
        fig.add_vline(x=median, line_dash="solid", line_color="green", annotation_text="Median")
        fig.add_vline(x=current_price, line_dash="solid", line_color="blue", annotation_text="Current Price")
        st.plotly_chart(fig)

else:
    st.error(f"Could not retrieve data for {ticker}. Please check the symbol and try again.")
