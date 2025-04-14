import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configure page
st.set_page_config(page_title="AI Portfolio Manager", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 1200px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
    }
    .stock-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Portfolio definition
PORTFOLIO = {
    'TCS.NS': 'Tata Consultancy Services',
    'ITC.NS': 'ITC Limited',
    'WIPRO.NS': 'Wipro Limited',
    'INFY.NS': 'Infosys Limited',
    'ABB.NS': 'ABB India Limited'
}

# Helper functions
def get_stock_data(ticker):
    """Fetch comprehensive stock data from yfinance"""
    stock = yf.Ticker(ticker)
    
    # Get current market data
    info = stock.info
    
    # Get historical data for technical analysis
    hist = stock.history(period="1y")
    
    return {
        'info': info,
        'history': hist
    }

def calculate_technical_indicators(df):
    """Calculate technical indicators from historical data"""
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_portfolio_metrics(portfolio_data):
    """Calculate portfolio-level metrics"""
    metrics = {
        'total_beta': 0,
        'total_pe': 0,
        'total_debt_to_equity': 0,
        'total_roe': 0,
        'sector_exposure': {},
        'correlation_matrix': None
    }
    
    # Prepare data for correlation matrix
    close_prices = pd.DataFrame()
    weights = []
    
    for ticker, data in portfolio_data.items():
        weight = 1/len(portfolio_data)  # Equal weighting for demo
        weights.append(weight)
        
        # Aggregate portfolio metrics (weighted)
        metrics['total_beta'] += data['info'].get('beta', 0) * weight
        metrics['total_pe'] += data['info'].get('trailingPE', 0) * weight
        metrics['total_debt_to_equity'] += data['info'].get('debtToEquity', 0) * weight
        metrics['total_roe'] += data['info'].get('returnOnEquity', 0) * weight
        
        # Track sector exposure
        sector = data['info'].get('sector', 'Unknown')
        metrics['sector_exposure'][sector] = metrics['sector_exposure'].get(sector, 0) + weight
        
        # Add to correlation matrix
        close_prices[ticker] = data['history']['Close']
    
    # Calculate correlation matrix
    if not close_prices.empty:
        metrics['correlation_matrix'] = close_prices.corr()
    
    return metrics

def generate_ai_insights(stock_data, portfolio_metrics):
    """Generate AI-powered insights based on the data"""
    insights = []
    warnings = []
    suggestions = []
    
    # Portfolio-level insights
    if portfolio_metrics['total_beta'] > 1.2:
        insights.append("Your portfolio has higher-than-market risk (Beta = {:.2f}). Consider adding defensive stocks.".format(portfolio_metrics['total_beta']))
    elif portfolio_metrics['total_beta'] < 0.8:
        insights.append("Your portfolio has lower-than-market risk (Beta = {:.2f}). You may be under-exposed to market upside.".format(portfolio_metrics['total_beta']))
    
    if portfolio_metrics['total_pe'] > 25:
        insights.append("Portfolio appears overvalued (Avg P/E = {:.1f}). Look for value opportunities.".format(portfolio_metrics['total_pe']))
    
    # Sector concentration warning
    if len(portfolio_metrics['sector_exposure']) < 3:
        main_sector = max(portfolio_metrics['sector_exposure'], key=portfolio_metrics['sector_exposure'].get)
        warnings.append(f"⚠️ High concentration in {main_sector} sector ({portfolio_metrics['sector_exposure'][main_sector]*100:.0f}%). Consider diversifying.")
    
    # Stock-specific analysis
    for ticker, data in stock_data.items():
        info = data['info']
        hist = data['history']
        
        # Valuation warnings
        if info.get('trailingPE', 0) > 30 and info.get('trailingPE', 0) > info.get('industryPE', 100):
            warnings.append(f"{ticker}: High P/E ratio ({info.get('trailingPE', 0):.1f}) compared to industry")
        
        # Financial health warnings
        if info.get('debtToEquity', 0) > 1.5:
            warnings.append(f"{ticker}: High debt-to-equity ratio ({info.get('debtToEquity', 0):.2f})")
        
        # Technical analysis signals
        last_rsi = hist['RSI'].iloc[-1]
        if last_rsi > 70:
            warnings.append(f"{ticker}: Overbought (RSI = {last_rsi:.1f})")
        elif last_rsi < 30:
            insights.append(f"{ticker}: Oversold (RSI = {last_rsi:.1f}) - potential buying opportunity")
        
        # Exit signal based on moving averages
        if hist['MA_50'].iloc[-1] < hist['MA_200'].iloc[-1] and hist['MA_50'].iloc[-2] >= hist['MA_200'].iloc[-2]:
            suggestions.append(f"🚨 Consider exiting {ticker} - Death Cross detected (50MA crossed below 200MA)")
    
    return {
        'insights': insights,
        'warnings': warnings,
        'suggestions': suggestions
    }

def detect_anomalies(portfolio_data):
    """Use Isolation Forest to detect anomalous behavior in stocks"""
    features = []
    tickers = []
    
    for ticker, data in portfolio_data.items():
        info = data['info']
        features.append([
            info.get('beta', 0),
            info.get('trailingPE', 0),
            info.get('debtToEquity', 0),
            info.get('returnOnEquity', 0),
            info.get('currentRatio', 0),
            info.get('quickRatio', 0),
            info.get('priceToBook', 0),
            data['history']['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        ])
        tickers.append(ticker)
    
    if not features:
        return {}
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    # Train Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    preds = clf.predict(X)
    
    # Return anomalous stocks
    anomalies = {}
    for i, pred in enumerate(preds):
        if pred == -1:
            anomalies[tickers[i]] = "Anomalous behavior detected across multiple metrics"
    
    return anomalies

# Main app
def main():
    st.title("📊 AI-Powered Portfolio Risk Dashboard")
    st.write(f"Analyzing your portfolio: {', '.join(PORTFOLIO.keys())}")
    
    # Load data
    with st.spinner("Fetching market data..."):
        portfolio_data = {ticker: get_stock_data(ticker) for ticker in PORTFOLIO}
        
        # Calculate technical indicators
        for ticker in portfolio_data:
            portfolio_data[ticker]['history'] = calculate_technical_indicators(portfolio_data[ticker]['history'])
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(portfolio_data)
        
        # Generate AI insights
        ai_output = generate_ai_insights(portfolio_data, portfolio_metrics)
        
        # Detect anomalies
        anomalies = detect_anomalies(portfolio_data)
    
    # Dashboard layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Beta", f"{portfolio_metrics['total_beta']:.2f}")
    with col2:
        st.metric("Avg P/E Ratio", f"{portfolio_metrics['total_pe']:.1f}")
    with col3:
        st.metric("Avg ROE", f"{portfolio_metrics['total_roe']:.2%}")
    
    # Risk summary
    st.subheader("🔍 AI-Generated Risk Assessment")
    
    if ai_output['warnings']:
        st.warning("### Risk Warnings")
        for warning in ai_output['warnings']:
            st.write(f"- {warning}")
    
    if anomalies:
        st.error("### Anomaly Detection Alerts")
        for ticker, message in anomalies.items():
            st.write(f"- {ticker}: {message}")
    
    if ai_output['insights']:
        st.info("### Insights")
        for insight in ai_output['insights']:
            st.write(f"- {insight}")
    
    if ai_output['suggestions']:
        st.success("### Actionable Suggestions")
        for suggestion in ai_output['suggestions']:
            st.write(f"- {suggestion}")
    
    # Portfolio analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Stock Details", "Sector Exposure", "Correlation Matrix", "Technical Analysis"])
    
    with tab1:
        st.subheader("Stock-Level Metrics")
        selected_ticker = st.selectbox("Select stock", list(PORTFOLIO.keys()), format_func=lambda x: f"{x} - {PORTFOLIO[x]}")
        
        if selected_ticker:
            data = portfolio_data[selected_ticker]
            info = data['info']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<div class='stock-header'>{selected_ticker} - {PORTFOLIO[selected_ticker]}</div>", unsafe_allow_html=True)
                st.write(f"Sector: {info.get('sector', 'N/A')}")
                st.write(f"Industry: {info.get('industry', 'N/A')}")
                
                st.metric("Current Price", f"₹{info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
                st.metric("52-Week Range", f"₹{info.get('fiftyTwoWeekLow', 'N/A')} - ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                
            with col2:
                st.markdown("### Valuation")
                pe_color = "risk-high" if info.get('trailingPE', 0) > 30 else "risk-medium" if info.get('trailingPE', 0) > 20 else "risk-low"
                st.markdown(f"P/E Ratio: <span class='{pe_color}'>{info.get('trailingPE', 'N/A')}</span>", unsafe_allow_html=True)
                
                pb_color = "risk-high" if info.get('priceToBook', 0) > 5 else "risk-medium" if info.get('priceToBook', 0) > 3 else "risk-low"
                st.markdown(f"Price-to-Book: <span class='{pb_color}'>{info.get('priceToBook', 'N/A')}</span>", unsafe_allow_html=True)
                
                st.markdown("### Financial Health")
                de_color = "risk-high" if info.get('debtToEquity', 0) > 1.5 else "risk-medium" if info.get('debtToEquity', 0) > 1 else "risk-low"
                st.markdown(f"Debt-to-Equity: <span class='{de_color}'>{info.get('debtToEquity', 'N/A')}</span>", unsafe_allow_html=True)
                
                cr_color = "risk-high" if info.get('currentRatio', 0) < 1 else "risk-medium" if info.get('currentRatio', 0) < 1.5 else "risk-low"
                st.markdown(f"Current Ratio: <span class='{cr_color}'>{info.get('currentRatio', 'N/A')}</span>", unsafe_allow_html=True)
            
            # Price chart
            st.subheader("Price Movement")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['history'].index, y=data['history']['Close'], name='Close Price'))
            fig.add_trace(go.Scatter(x=data['history'].index, y=data['history']['MA_50'], name='50-Day MA'))
            fig.add_trace(go.Scatter(x=data['history'].index, y=data['history']['MA_200'], name='200-Day MA'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sector Exposure")
        if portfolio_metrics['sector_exposure']:
            fig = go.Figure(go.Pie(
                labels=list(portfolio_metrics['sector_exposure'].keys()),
                values=list(portfolio_metrics['sector_exposure'].values()),
                hole=0.3
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sector data available")
    
    with tab3:
        st.subheader("Correlation Matrix")
        if portfolio_metrics['correlation_matrix'] is not None:
            fig = go.Figure(go.Heatmap(
                z=portfolio_metrics['correlation_matrix'],
                x=portfolio_metrics['correlation_matrix'].columns,
                y=portfolio_metrics['correlation_matrix'].columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("""
            **Interpretation:**
            - Values close to 1 indicate high positive correlation (stocks move together)
            - Values close to -1 indicate negative correlation (stocks move opposite)
            - Ideal portfolio has some low/negative correlations for diversification
            """)
        else:
            st.warning("Could not calculate correlation matrix")
    
    with tab4:
        st.subheader("Technical Indicators")
        selected_ticker_ta = st.selectbox("Select stock for technical analysis", list(PORTFOLIO.keys()), format_func=lambda x: f"{x} - {PORTFOLIO[x]}", key='ta_select')
        
        if selected_ticker_ta:
            data = portfolio_data[selected_ticker_ta]
            hist = data['history']
            
            # Create subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Price and moving averages
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_50'], name='50-Day MA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA_200'], name='200-Day MA'), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Current technical signals
            last_rsi = hist['RSI'].iloc[-1]
            ma_signal = "Bullish (50MA > 200MA)" if hist['MA_50'].iloc[-1] > hist['MA_200'].iloc[-1] else "Bearish (50MA ≤ 200MA)"
            
            st.write(f"""
            **Current Signals:**
            - RSI: {last_rsi:.1f} ({'Overbought' if last_rsi > 70 else 'Oversold' if last_rsi < 30 else 'Neutral'})
            - Moving Averages: {ma_signal}
            """)

if __name__ == "__main__":
    main()
