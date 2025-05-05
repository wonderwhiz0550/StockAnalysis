import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib as ta
from configparser import ConfigParser

# Set page title and layout
st.set_page_config(page_title="Stock Analysis Tool", layout="wide")

# Load configuration
config = ConfigParser()
config.read('config.ini')

def get_config_value(section, key, default=None):
    try:
        return config.get(section, key)
    except:
        return default

# App title and description
st.title('Stock Analysis Tool')
st.write('Analyze stocks with technical indicators and Monte Carlo simulations')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Main Settings", "Technical Analysis", "Monte Carlo Simulation"])

# Main Settings Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Ticker symbol
        default_ticker = get_config_value('General', 'default_ticker', 'AAPL')
        ticker = st.text_input('Enter Stock Ticker Symbol', value=default_ticker)
        
        # Time period
        default_period = get_config_value('General', 'default_period', '1y')
        period_options = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        period = st.selectbox('Select Time Period', period_options, 
                             index=period_options.index(default_period) if default_period in period_options else 3)
        
        # Analysis selection
        st.subheader("Select Analysis to Perform")
        do_technical = st.checkbox("Technical Analysis", value=True)
        do_monte_carlo = st.checkbox("Monte Carlo Simulation", value=True)
    
    with col2:
        # Interval
        default_interval = get_config_value('General', 'default_interval', '1d')
        interval_options = ['1d', '5d', '1wk', '1mo', '3mo']
        interval = st.selectbox('Select Interval', interval_options,
                              index=interval_options.index(default_interval) if default_interval in interval_options else 0)
        
        # Future days prediction (for Monte Carlo)
        default_future_days = int(get_config_value('MonteCarlo', 'future_days', '30'))
        future_days = st.number_input('Days to predict in the future', 
                                    min_value=1, max_value=365, value=default_future_days)

# Technical Analysis Tab
with tab2:
    if do_technical:
        st.header("Technical Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Technical Indicators - Moving Averages
            st.subheader("Moving Averages")
            default_ma_short = int(get_config_value('TechnicalAnalysis', 'ma_short', '20'))
            default_ma_medium = int(get_config_value('TechnicalAnalysis', 'ma_medium', '50'))
            default_ma_long = int(get_config_value('TechnicalAnalysis', 'ma_long', '200'))
            
            ma_short = st.number_input('Short MA period', min_value=1, max_value=100, value=default_ma_short)
            ma_medium = st.number_input('Medium MA period', min_value=1, max_value=200, value=default_ma_medium)
            ma_long = st.number_input('Long MA period', min_value=1, max_value=500, value=default_ma_long)
        
        with col2:
            # Technical Indicators - RSI, MACD
            st.subheader("Oscillators")
            default_rsi_period = int(get_config_value('TechnicalAnalysis', 'rsi_period', '14'))
            rsi_period = st.number_input('RSI Period', min_value=1, max_value=50, value=default_rsi_period)
            
            default_macd_fast = int(get_config_value('TechnicalAnalysis', 'macd_fast', '12'))
            default_macd_slow = int(get_config_value('TechnicalAnalysis', 'macd_slow', '26'))
            default_macd_signal = int(get_config_value('TechnicalAnalysis', 'macd_signal', '9'))
            
            macd_fast = st.number_input('MACD Fast Period', min_value=1, max_value=50, value=default_macd_fast)
            macd_slow = st.number_input('MACD Slow Period', min_value=1, max_value=100, value=default_macd_slow)
            macd_signal = st.number_input('MACD Signal Period', min_value=1, max_value=50, value=default_macd_signal)
    else:
        st.info("Technical Analysis is disabled. Enable it in the Main Settings tab.")

# Monte Carlo Tab
with tab3:
    if do_monte_carlo:
        st.header("Monte Carlo Simulation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_simulations = int(get_config_value('MonteCarlo', 'simulations', '1000'))
            simulations = st.number_input('Number of Simulations', 
                                        min_value=100, max_value=10000, value=default_simulations)
        
        with col2:
            default_confidence_level = float(get_config_value('MonteCarlo', 'confidence_level', '0.95'))
            confidence_level = st.slider('Confidence Level', 
                                       min_value=0.5, max_value=0.99, value=default_confidence_level, step=0.01)
    else:
        st.info("Monte Carlo Simulation is disabled. Enable it in the Main Settings tab.")

# Run Analysis Button
if st.button('Run Analysis'):
    if ticker:
        try:
            # Download stock data
            stock_data = yf.download(ticker, period=period, interval=interval)
            
            if stock_data.empty:
                st.error("No data found for the specified ticker or period.")
            else:
                # Display basic information
                st.header(f"{ticker} Stock Analysis")
                st.write(f"Period: {period}, Interval: {interval}")
                
                # Create tabs for results
                result_tabs = []
                
                if do_technical:
                    result_tabs.append("Technical Analysis")
                
                if do_monte_carlo:
                    result_tabs.append("Monte Carlo Simulation")
                
                result_tabs.append("Raw Data")
                
                results = st.tabs(result_tabs)
                
                tab_index = 0
                
                # Technical Analysis
                if do_technical:
                    with results[tab_index]:
                        st.subheader("Technical Analysis")
                        
                        # Calculate technical indicators
                        df = stock_data.copy()
                        
                        # Moving Averages
                        df[f'MA{ma_short}'] = df['Close'].rolling(ma_short).mean()
                        df[f'MA{ma_medium}'] = df['Close'].rolling(ma_medium).mean()
                        df[f'MA{ma_long}'] = df['Close'].rolling(ma_long).mean()
                        
                        # RSI
                        df['RSI'] = ta.RSI(df['Close'], timeperiod=rsi_period)
                        
                        # MACD
                        macd, macd_signal, macd_hist = ta.MACD(df['Close'], 
                                                              fastperiod=macd_fast, 
                                                              slowperiod=macd_slow, 
                                                              signalperiod=macd_signal)
                        df['MACD'] = macd
                        df['MACD_Signal'] = macd_signal
                        df['MACD_Hist'] = macd_hist
                        
                        # Create plots
                        # Price and Moving Averages
                        fig = make_subplots(rows=3, cols=1, 
                                           shared_xaxes=True,
                                           vertical_spacing=0.05,
                                           row_heights=[0.5, 0.25, 0.25],
                                           subplot_titles=('Price and Moving Averages', 'RSI', 'MACD'))
                        
                        # Candlestick plot
                        fig.add_trace(
                            go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )
                        
                        # Moving averages
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df[f'MA{ma_short}'], 
                                      name=f'MA{ma_short}', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df[f'MA{ma_medium}'], 
                                      name=f'MA{ma_medium}', line=dict(color='orange')),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df[f'MA{ma_long}'], 
                                      name=f'MA{ma_long}', line=dict(color='green')),
                            row=1, col=1
                        )
                        
                        # RSI
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                            row=2, col=1
                        )
                        
                        # Add RSI overbought/oversold lines
                        fig.add_trace(
                            go.Scatter(x=df.index, y=[70] * len(df), name='Overbought', 
                                      line=dict(color='red', dash='dash')),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df.index, y=[30] * len(df), name='Oversold', 
                                      line=dict(color='green', dash='dash')),
                            row=2, col=1
                        )
                        
                        # MACD
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
                            row=3, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
                            row=3, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color='green'),
                            row=3, col=1
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{ticker} Technical Analysis',
                            height=900,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show signals
                        st.subheader("Technical Signals")
                        
                        # Current values
                        current_price = df['Close'].iloc[-1]
                        current_ma_short = df[f'MA{ma_short}'].iloc[-1]
                        current_ma_medium = df[f'MA{ma_medium}'].iloc[-1]
                        current_ma_long = df[f'MA{ma_long}'].iloc[-1]
                        current_rsi = df['RSI'].iloc[-1]
                        current_macd = df['MACD'].iloc[-1]
                        current_macd_signal = df['MACD_Signal'].iloc[-1]
                        
                        # Signal columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                            st.metric(f"MA {ma_short}", f"${current_ma_short:.2f}", 
                                     f"{((current_price/current_ma_short)-1)*100:.2f}%")
                            st.metric(f"MA {ma_medium}", f"${current_ma_medium:.2f}", 
                                     f"{((current_price/current_ma_medium)-1)*100:.2f}%")
                            st.metric(f"MA {ma_long}", f"${current_ma_long:.2f}", 
                                     f"{((current_price/current_ma_long)-1)*100:.2f}%")
                        
                        with col2:
                            # RSI Signal
                            rsi_signal = "NEUTRAL"
                            if current_rsi > 70:
                                rsi_signal = "OVERBOUGHT"
                            elif current_rsi < 30:
                                rsi_signal = "OVERSOLD"
                            
                            st.metric("RSI", f"{current_rsi:.2f}", rsi_signal)
                            
                            # MACD Signal
                            macd_signal_text = "NEUTRAL"
                            if current_macd > current_macd_signal:
                                macd_signal_text = "BULLISH"
                            else:
                                macd_signal_text = "BEARISH"
                            
                            st.metric("MACD", f"{current_macd:.4f}", macd_signal_text)
                            st.metric("MACD Signal", f"{current_macd_signal:.4f}")
                        
                        with col3:
                            # Moving Average signals
                            ma_short_medium_signal = "NEUTRAL"
                            if current_ma_short > current_ma_medium:
                                ma_short_medium_signal = "BULLISH"
                            else:
                                ma_short_medium_signal = "BEARISH"
                            
                            ma_medium_long_signal = "NEUTRAL"
                            if current_ma_medium > current_ma_long:
                                ma_medium_long_signal = "BULLISH"
                            else:
                                ma_medium_long_signal = "BEARISH"
                            
                            st.metric(f"MA{ma_short} vs MA{ma_medium}", ma_short_medium_signal)
                            st.metric(f"MA{ma_medium} vs MA{ma_long}", ma_medium_long_signal)
                    
                    tab_index += 1
                
                # Monte Carlo Simulation
                if do_monte_carlo:
                    with results[tab_index]:
                        st.subheader("Monte Carlo Simulation")
                        
                        # Calculate daily returns
                        df = stock_data.copy()
                        df['Daily Return'] = df['Close'].pct_change()
                        
                        # Get mean and standard deviation of daily returns
                        mu = df['Daily Return'].mean()
                        sigma = df['Daily Return'].std()
                        
                        # Get last price
                        last_price = df['Close'].iloc[-1]
                        
                        # Run simulation
                        simulation_df = pd.DataFrame()
                        
                        for i in range(simulations):
                            # Create list of daily returns with same mean and std
                            daily_returns = np.random.normal(mu, sigma, future_days) + 1
                            
                            # Start price is the last price
                            price_series = [last_price]
                            
                            # Calculate price for each day
                            for x in daily_returns:
                                price_series.append(price_series[-1] * x)
                            
                            # Store simulation
                            simulation_df[i] = price_series
                        
                        # Plot
                        fig = go.Figure()
                        
                        # Add traces for each simulation run
                        for i in range(min(50, simulations)):  # Only plot 50 lines for visibility
                            fig.add_trace(
                                go.Scatter(
                                    y=simulation_df[i],
                                    mode='lines',
                                    line=dict(width=0.5, color='rgba(100, 100, 100, 0.2)'),
                                    showlegend=False
                                )
                            )
                        
                        # Add trace for the mean
                        mean_simulation = simulation_df.mean(axis=1)
                        fig.add_trace(
                            go.Scatter(
                                y=mean_simulation,
                                mode='lines',
                                line=dict(width=2, color='blue'),
                                name='Mean'
                            )
                        )
                        
                        # Add confidence interval
                        upper_ci = []
                        lower_ci = []
                        
                        for i in range(len(mean_simulation)):
                            upper = np.percentile(simulation_df.iloc[i], 100 * confidence_level)
                            lower = np.percentile(simulation_df.iloc[i], 100 * (1 - confidence_level))
                            upper_ci.append(upper)
                            lower_ci.append(lower)
                        
                        fig.add_trace(
                            go.Scatter(
                                y=upper_ci,
                                mode='lines',
                                line=dict(width=1, color='red'),
                                name=f'Upper {confidence_level*100}% CI'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                y=lower_ci,
                                mode='lines',
                                line=dict(width=1, color='red'),
                                name=f'Lower {confidence_level*100}% CI',
                                fill='tonexty'
                            )
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{ticker} Monte Carlo Simulation ({future_days} days)',
                            xaxis_title='Days',
                            yaxis_title='Price',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        st.subheader("Prediction Statistics")
                        
                        # Calculate statistics for the final day
                        final_day = simulation_df.iloc[-1]
                        
                        expected_price = round(final_day.mean(), 2)
                        ci_lower = round(np.percentile(final_day, 100 * (1 - confidence_level)), 2)
                        ci_upper = round(np.percentile(final_day, 100 * confidence_level), 2)
                        
                        price_change = round(((expected_price / last_price) - 1) * 100, 2)
                        max_price = round(final_day.max(), 2)
                        min_price = round(final_day.min(), 2)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Starting Price", f"${last_price:.2f}")
                            st.metric("Expected Price", f"${expected_price:.2f}", f"{price_change}%")
                        
                        with col2:
                            st.metric(f"Lower {confidence_level*100}% CI", f"${ci_lower:.2f}", 
                                    f"{round(((ci_lower/last_price)-1)*100, 2)}%")
                            st.metric(f"Upper {confidence_level*100}% CI", f"${ci_upper:.2f}", 
                                    f"{round(((ci_upper/last_price)-1)*100, 2)}%")
                        
                        with col3:
                            st.metric("Maximum Price", f"${max_price:.2f}", 
                                     f"{round(((max_price/last_price)-1)*100, 2)}%")
                            st.metric("Minimum Price", f"${min_price:.2f}", 
                                     f"{round(((min_price/last_price)-1)*100, 2)}%")
                    
                    tab_index += 1
                
                # Raw Data tab
                with results[tab_index]:
                    st.subheader("Raw Stock Data")
                    st.dataframe(stock_data)
                    
                    # Download option
                    csv = stock_data.to_csv()
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"{ticker}_stock_data.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid ticker symbol.")