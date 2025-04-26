# --- app_streamlit.py ---

import streamlit as st
import plotly.graph_objs as go
import pandas as pd
from ExpectationInvesting_Code import (
    evaluate_stock,
    config,
    calculate_technical_indicators,
    fetch_analyst_ratings,
    fetch_news_sentiment,
    buy_sell_hold_logic
)

# Streamlit App Config
st.set_page_config(page_title="Stock Decision Helper", layout="wide")
st.title("📈 Enhanced Stock Valuation & Recommendation Tool")

# User Inputs
ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
st.markdown("### Select Analysis Modules")

do_monte_carlo = st.checkbox("Run Monte Carlo Simulation", value=True)
do_technical = st.checkbox("Show Technical Indicators (RSI, SMA, MACD, Bollinger Bands)")
do_analyst = st.checkbox("Fetch Analyst Ratings")
do_news = st.checkbox("Fetch News Sentiment")
do_recommendation = st.checkbox("Generate Final Buy/Sell/Hold Recommendation", value=True)

if st.button("🚀 Run Analysis"):
    st.subheader(f"Results for {ticker}")

    # Monte Carlo Simulation
    if do_monte_carlo:
        result, status, plot_path = evaluate_stock(ticker, config)
        if status != "Success":
            st.error(status)
        else:
            st.metric("Current Price", f"${result['stock_price']:.2f}")
            st.metric("Implied Value", f"${result['mean_simulated_price']:.2f}")
            st.image(plot_path, caption="Monte Carlo Simulation")

    # Technical Indicators
    if do_technical:
        tech_df = calculate_technical_indicators(ticker)
        if not tech_df.empty:
            # Price and Moving Averages Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['SMA50'], mode='lines', name='SMA50'))
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['SMA200'], mode='lines', name='SMA200'))
            if 'Upper_BB' in tech_df.columns and 'Lower_BB' in tech_df.columns:
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Upper_BB'], mode='lines', name='Upper BB', line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Lower_BB'], mode='lines', name='Lower BB', line=dict(dash='dot')))
            fig.update_layout(title=f"{ticker} Price with Moving Averages and Bollinger Bands",
                              xaxis_title="Date", yaxis_title="Price",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- FIX RSI and MACD extraction ---
            rsi_series = tech_df['RSI'].squeeze()
            macd_series = tech_df['MACD'].squeeze()

            # --- ADD THESE LINES FOR DEBUGGING ---
            st.write("Shape of RSI Series:", rsi_series.shape)
            st.write("Type of RSI Series:", type(rsi_series))
            st.write("First 5 values of RSI Series:", rsi_series.head())
            st.write("Shape of MACD Series:", macd_series.shape)
            st.write("Type of MACD Series:", type(macd_series))
            st.write("First 5 values of MACD Series:", macd_series.head())
            # --- END DEBUGGING LINES ---

            st.line_chart(rsi_series.dropna())
            st.line_chart(macd_series.dropna())

    # Analyst Ratings
    if do_analyst:
        ratings = fetch_analyst_ratings(ticker)
        if not ratings.empty:
            st.dataframe(ratings)

    # News Sentiment
    if do_news:
        news = fetch_news_sentiment(ticker)
        for headline, sentiment in news:
            st.write(f"**{sentiment}**: {headline}")

    # Final Buy/Sell/Hold Recommendation
    if do_recommendation and do_monte_carlo:
        latest_rsi = rsi_series.dropna().iloc[-1] if do_technical and not tech_df.empty else 50
        analyst_text = "; ".join(ratings['To Grade']) if do_analyst and not ratings.empty else "Hold"
        news_text = "; ".join(s for _, s in news) if do_news else "Neutral"
        decision = buy_sell_hold_logic(result['stock_price'], result['mean_simulated_price'], analyst_text, news_text, latest_rsi)
        st.success(f"📢 Final Recommendation: **{decision}**")
