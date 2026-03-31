import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import numpy as np
from datetime import date

# --- Configuration ---
st.set_page_config(page_title="Algorithmic Strategy Optimizer", layout="wide")

# --- Helper Functions ---
def fetch_data_from_yahoo(ticker_symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        df.index = df.index.tz_localize(None)
        df.reset_index(inplace=True)
        if 'Dividends' not in df.columns: df['Dividends'] = 0.0
        return df, ticker.info
    except Exception as e:
        return None, str(e)

def xirr(transactions):
    if not transactions: return 0.0
    transactions.sort(key=lambda x: x[0])
    amounts = [t[1] for t in transactions]
    if all(a >= 0 for a in amounts) or all(a <= 0 for a in amounts): return 0.0
    start_date = transactions[0][0]
    
    def npv(rate):
        total_npv = 0.0
        for dt, amt in transactions:
            days = (dt - start_date).days
            if rate <= -1.0: rate = -0.9999
            total_npv += amt / ((1 + rate) ** (days / 365.0))
        return total_npv

    def npv_derivative(rate):
        d_npv = 0.0
        for dt, amt in transactions:
            days = (dt - start_date).days
            if rate <= -1.0: rate = -0.9999
            d_npv -= (days / 365.0) * amt / ((1 + rate) ** ((days / 365.0) + 1))
        return d_npv

    rate = 0.1
    for _ in range(50):
        try:
            n_val = npv(rate)
            d_val = npv_derivative(rate)
            if abs(d_val) < 1e-6: break
            new_rate = rate - n_val / d_val
            if abs(new_rate - rate) < 1e-6: return new_rate
            rate = new_rate
        except: return 0.0
    return rate

def run_simulation(df_raw, params):
    # Prepare Data with Indicators
    df = df_raw.copy()
    
    # Calculate Custom Indicators based on params
    if params['strategy_mode'] == "SMA Crossover":
        df['SMA_S'] = df['Close'].rolling(window=params['sma_short']).mean()
        df['SMA_L'] = df['Close'].rolling(window=params['sma_long']).mean()
    elif params['strategy_mode'] == "RSI Mean Reversion":
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    elif params['strategy_mode'] == "Trend-Filtered Dip":
        df['SMA_Trend'] = df['Close'].rolling(window=params['trend_sma']).mean()

    # Unpack parameters
    strategy_mode = params['strategy_mode']
    
    # Strategy Specific Params
    buy_drop_pct = params.get('buy_drop_pct', 0)
    sell_profit_pct = params.get('sell_profit_pct', 0)
    use_trailing_stop = params.get('use_trailing_stop', False)
    trailing_stop_pct = params.get('trailing_stop_pct', 0)
    
    rsi_buy = params.get('rsi_buy', 30)
    rsi_sell = params.get('rsi_sell', 70)
    
    trend_sma = params.get('trend_sma', 200)
    confirmation_days = params.get('confirmation_days', 3)
    
    interest_rate_pct = params['interest_rate_pct']
    enable_dividends = params['enable_dividends']
    restrict_ex_date = params['restrict_ex_date']
    
    initial_investment = params['initial_investment']
    monthly_investment = params['monthly_investment']
    
    trade_size_type = params['trade_size_type']
    shares_per_trade = params['shares_per_trade']
    min_trade_amt = params['min_trade_amt']
    max_trade_amt = params['max_trade_amt']

    # --- Pre-Calculation: Restricted Days ---
    restricted_indices = set()
    dividend_events = []
    
    if enable_dividends:
        div_indices = df.index[df['Dividends'] > 0].tolist()
        for idx in div_indices:
            dividend_events.append({"Date": df.loc[idx, 'Date'], "Amount": df.loc[idx, 'Dividends']})
            if restrict_ex_date:
                restricted_indices.add(idx)
                if idx > 0: restricted_indices.add(idx - 1)
                if idx < len(df) - 1: restricted_indices.add(idx + 1)

    # --- 1. Signal Generation ---
    potential_trades = []
    trades_by_date = {} 
    
    for i in range(1, len(df)):
        daily_date = df.loc[i, 'Date']
        if daily_date not in trades_by_date:
            trades_by_date[daily_date] = {'buys': [], 'sells': []}
        
        if i in restricted_indices: continue 
        
        # --- STRATEGY LOGIC
