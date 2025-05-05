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

# Import the ExpectationIvesting_code module
try:
    from ExpectationIvesting_Code import get_data, monte_carlo_simulation, calculate_confidence_interval, analyze_returns
except ImportError:
    st.error("Error importing ExpectationIvesting_code module. Make sure the file is in the same directory.")
    
    # Define fallback functions in case import fails
    def get_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        return data
    
    def monte_carlo_simulation(ticker_data, num_simulations, time_horizon, last_price=None):
        returns = ticker_data['Adj Close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        if last_price is None:
            last_price = ticker_data['Adj Close'].iloc[-1]
        simulation_df = pd.DataFrame()
        for i in range(num_simulations):
            prices = [last_price]
            for _ in range(time_horizon):
                daily_return = np.random.normal(mu, sigma)
                next_price = prices[-1] * (1 + daily_return)
                prices.append(next_price)
            simulation_df[i] = prices
        final_prices = simulation_df.iloc[-1]
        metrics = {'mean': final_prices.mean(), 'median': final_prices.median()}
        return simulation_df, metrics
    
    def calculate_confidence_interval(simulation_df, confidence_level=95):
        lower_percentile = (100 - confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        final_values = simulation_df.iloc[-1]
        lower_bound = final_values.quantile(lower_percentile / 100)
        upper_bound = final_values.quantile(upper_percentile / 100)
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'median': final_values.median(),
            'confidence_level': confidence_level
        }
    
    def analyze_returns(simulation_df, current_price):
        final_prices = simulation_df.iloc[-1]
        returns = (final_prices - current_price) / current_price * 100
        metrics = {'mean_return': returns.mean(), 'median_return': returns.median()}
        return metrics

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

# Load data based on inputs
try:
    if use_custom_dates:
        data = get_data(ticker, start_date, end_date)
    else:
        # Convert period to start_date, end_date
        end_date = date.today()
        if period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '2y':
            start_date = end_date - timedelta(days=365*2)
        elif period == '5y':
            start_date = end_date - timedelta(days=365*5)
        elif period == '10y':
            start_date = end_date - timedelta(days=365*10)
        elif period == 'ytd':
            start_date = date(end_date.year, 1, 1)
        else:  # 'max'
            start_date = end_date - timedelta(days=365*20)  # Use 20 years as max
            
        data = get_data(ticker, start_date, end_date)

    # If data is available, display analysis
    if not data.empty:
        # Display basic stock information
        st.header(f'{ticker} Stock Analysis')
        
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
            
            with st.spinner('Running simulations...'):
                # Run simulation
                simulation_df, sim_metrics = monte_carlo_simulation(data, num_simulations, num_days)
                
                # Calculate confidence intervals
                ci = calculate_confidence_interval(simulation_df, confidence_level)
                ci['confidence_level'] = confidence_level
                
                # Analyze returns
                return_metrics = analyze_returns(simulation_df, current_price)
                
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
                
                # Create future dates
                future_dates = pd.date_range(start=data.index[-1], periods=len(simulation_df), freq='B')
                
                # Add a sample of Monte Carlo simulations (first 100 only for visual clarity)
                for i in range(min(50, num_simulations)):
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=simulation_df[i],
                            mode='lines',
                            opacity=0.1,
                            line=dict(color='green'),
                            name=f'Simulation {i}',
                            showlegend=False
                        )
                    )
                
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
                col1.metric("Expected Price", f"${ci['median']:.2f}")
                col2.metric(f"Lower {confidence_level}% Bound", f"${ci['lower_bound']:.2f}")
                col3.metric(f"Upper {confidence_level}% Bound", f"${ci['upper_bound']:.2f}")
                
                # Calculate returns at different percentiles
                expected_return = (ci['median'] - current_price) / current_price * 100
                lower_return = (ci['lower_bound'] - current_price) / current_price * 100
                upper_return = (ci['upper_bound'] - current_price) / current_price * 100
                
                st.write(f"Expected return: {expected_return:.2f}%")
                st.write(f"Return range ({confidence_level}% confidence): {lower_return:.2f}% to {upper_return:.2f}%")
                
                # Histogram of final prices
                st.subheader('Distribution of Final Simulated Prices')
                fig = px.histogram(final_price, nbins=50, title=f'Histogram of {ticker} Simulated Prices after {num_days} Days')
                fig.add_vline(x=ci['lower_bound'], line_dash="dash", line_color="red", annotation_text=f"Lower {confidence_level}% CI")
                fig.add_vline(x=ci['upper_bound'], line_dash="dash", line_color="red", annotation_text=f"Upper {confidence_level}% CI")
                fig.add_vline(x=ci['median'], line_dash="solid", line_color="green", annotation_text="Median")
                fig.add_vline(x=current_price, line_dash="solid", line_color="blue", annotation_text="Current Price")
                st.plotly_chart(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your inputs and try again.")
