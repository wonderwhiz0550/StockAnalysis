import streamlit as st
from ExpectationInvesting_Code import evaluate_stock, config
import yfinance as yf
import pandas as pd
import io
from fpdf import FPDF

st.set_page_config(page_title="Stock Valuation Tool", layout="wide")

# -- Custom CSS omitted for brevity (keep your existing styles) --

# Helper: number formatting

def format_large_number(number):
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    elif abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    else:
        return f"${number:,.0f}"

# Helper: fetch peer multiples

def fetch_peer_multiples(peers):
    records = []
    for t in peers:
        tk = yf.Ticker(t.strip().upper())
        info = tk.info
        pe = info.get('trailingPE', None)
        ev = info.get('enterpriseValue', None)
        ebitda = info.get('ebitda', None)
        ev_ebitda = ev / ebitda if ev and ebitda else None
        records.append({
            'Ticker': t.strip().upper(),
            'P/E': pe,
            'EV/EBITDA': ev_ebitda
        })
    return pd.DataFrame(records)

# Helpers: generate Excel and PDF

def generate_excel(result, peers_df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    # write valuation result
    pd.DataFrame([{
        'Current Price': result['stock_price'],
        'Implied Price': result['mean_simulated_price'],
        'Valuation': result['valuation_status'],
        'FCF Margin': result['fcf_margin']
    }]).to_excel(writer, index=False, sheet_name='Valuation')
    # write peers
    peers_df.to_excel(writer, index=False, sheet_name='Peers')
    writer.save()
    output.seek(0)
    return output


def generate_pdf(result, peers_df, plot_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Valuation Report", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    # Metrics
    pdf.cell(0, 8, f"Current Price: ${result['stock_price']:.2f}", ln=True)
    pdf.cell(0, 8, f"Implied Price: ${result['mean_simulated_price']:.2f}", ln=True)
    pdf.cell(0, 8, f"Valuation Status: {result['valuation_status']}", ln=True)
    pdf.cell(0, 8, f"FCF Margin: {result['fcf_margin']:.2%}", ln=True)
    pdf.ln(5)
    # Insert histogram
    pdf.image(plot_path, w=150)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Peer Multiples", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", size=11)
    # Table header
    col_w = [40, 40, 40]
    headers = ['Ticker', 'P/E', 'EV/EBITDA']
    for i, h in enumerate(headers): pdf.cell(col_w[i], 8, h, border=1)
    pdf.ln()
    # Table rows
    for _, row in peers_df.iterrows():
        pdf.cell(col_w[0], 8, str(row['Ticker']), border=1)
        pdf.cell(col_w[1], 8, f"{row['P/E']:.2f}" if row['P/E'] else 'N/A', border=1)
        pdf.cell(col_w[2], 8, f"{row['EV/EBITDA']:.2f}" if row['EV/EBITDA'] else 'N/A', border=1)
        pdf.ln()
    buf = io.BytesIO(pdf.output(dest='S').encode('latin-1'))
    buf.seek(0)
    return buf

# Title
tabs = st.tabs(["🏢 Stock", "📈 Growth", "💸 Discount", "🎲 Simulation", "📊 Benchmark & Export"])

# Inputs same as before... (tabs[0] to tabs[3] code)
# ...

# New tab: Benchmark & Export
with tabs[4]:
    peers_input = st.text_area("Peer Tickers (comma-separated)", value="AAPL, GOOGL, AMZN")
    if st.button("🔍 Fetch Benchmarks"):
        peers = [p.strip() for p in peers_input.split(',') if p.strip()]
        peers_df = fetch_peer_multiples(peers)
        st.dataframe(peers_df)
    st.markdown("---")
    if st.button("🚀 Run Valuation & Fetch Benchmarks"):
        # Run valuation
        result, status, plot_path = evaluate_stock(ticker.upper(), config)
        if status != "Success":
            st.error(status)
        else:
            # Fetch peers
            peers = [p.strip() for p in peers_input.split(',') if p.strip()]
            peers_df = fetch_peer_multiples(peers)
            st.success("✅ Done")
            # Display metrics and chart (as before)
            cols = st.columns(3)
            with cols[0]:
                st.metric("Current Price", f"${result['stock_price']:.2f}")
                st.metric("Implied", f"${result['mean_simulated_price']:.2f}")
            with cols[1]:
                st.metric("FCF Margin", f"{result['fcf_margin']:.2%}")
            with cols[2]:
                st.metric("Valuation", result['valuation_status'])
            st.image(plot_path, use_container_width=True)
            st.dataframe(peers_df)
            # Export buttons
            excel_buf = generate_excel(result, peers_df)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buf,
                file_name=f"{ticker.upper()}_valuation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            pdf_buf = generate_pdf(result, peers_df, plot_path)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buf,
                file_name=f"{ticker.upper()}_valuation.pdf",
                mime="application/pdf"
            )
