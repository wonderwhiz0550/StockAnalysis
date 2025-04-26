# Updated Enhanced Modular Design with Interactive Charts and Correct Indentation

# --- app_streamlit.py --- FRONTEND ---

import streamlit as st
import plotly.graph_objs as go
from ExpectationInvesting_Code import (
    evaluate_stock,
    config,
    calculate_technical_indicators,
    fetch_analyst_ratings,
    fetch_news_sentiment,
    buy_sell_hold_logic
)

st.set_page_config(page_title="Stock Decision Helper", layout="wide")
st.title("📈 Enhanced Stock Valuation & Recommendation Tool")

ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
st.markdown("### Select Analysis Modules")

do_monte_carlo = st.checkbox("Run Monte Carlo Simulation", value=True)
do_technical = st.checkbox("Show Technical Indicators (RSI, SMA, MACD, Bollinger Bands)")
do_analyst = st.checkbox("Fetch Analyst Ratings")
do_news = st.checkbox("Fetch News Sentiment")
do_recommendation = st.checkbox("Generate Final Buy/Sell/Hold Recommendation", value=True)

if st.button("🚀 Run Analysis"):
    st.subheader(f"Results for {ticker}")

    if do_monte_carlo:
        result, status, plot_path = evaluate_stock(ticker, config)
        if status != "Success":
            st.error(status)
        else:
            st.metric("Current Price", f"${result['stock_price']:.2f}")
            st.metric("Implied Value", f"${result['mean_simulated_price']:.2f}")
            st.image(plot_path, caption="Monte Carlo Simulation")

    if do_technical:
        tech_df = calculate_technical_indicators(ticker)
        if not tech_df.empty:
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

            st.line_chart(tech_df['RSI'].dropna().squeeze())
            st.line_chart(tech_df['MACD'].dropna().squeeze())

    if do_analyst:
        ratings = fetch_analyst_ratings(ticker)
        if not ratings.empty:
            st.dataframe(ratings)

    if do_news:
        news = fetch_news_sentiment(ticker)
        for headline, sentiment in news:
            st.write(f"**{sentiment}**: {headline}")

    if do_recommendation and do_monte_carlo:
        latest_rsi = tech_df['RSI'].dropna().iloc[-1] if do_technical and not tech_df.empty else 50
        analyst_text = "; ".join(ratings['To Grade']) if do_analyst and not ratings.empty else "Hold"
        news_text = "; ".join(s for _, s in news) if do_news else "Neutral"
        decision = buy_sell_hold_logic(result['stock_price'], result['mean_simulated_price'], analyst_text, news_text, latest_rsi)
        st.success(f"📢 Final Recommendation: **{decision}**")

# --- ExpectationInvesting_Code.py --- BACKEND ---

# Functions remain unchanged.
