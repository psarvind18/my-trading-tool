import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
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

def run_simulation(df, params):
    # Unpack parameters
    strategy_mode = params['strategy_mode']
    buy_drop_pct = params['buy_drop_pct']
    sell_profit_pct = params['sell_profit_pct']
    use_trailing_stop = params['use_trailing_stop']
    trailing_stop_pct = params['trailing_stop_pct']
    
    interest_rate_pct = params['interest_rate_pct']
    enable_dividends = params['enable_dividends']
    restrict_ex_date = params['restrict_ex_date']
    
    initial_investment = params['initial_investment']
    monthly_investment = params['monthly_investment']
    shares_per_trade = params['shares_per_trade']

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
        
        prev_close = df.loc[i-1, 'Close']
        daily_low = df.loc[i, 'Low']
        
        drop_level = 1
        while True:
            current_drop_pct = buy_drop_pct * drop_level
            target_buy_price = prev_close * (1 - (current_drop_pct / 100.0))
            
            if daily_low <= target_buy_price:
                buy_price = target_buy_price
                sell_date = pd.NaT
                sell_price = 0.0
                status = "Open"
                
                if strategy_mode == "Swing Trading":
                    target_activation_price = buy_price * (1 + (sell_profit_pct / 100.0))
                    
                    if not use_trailing_stop:
                        # Fixed Target
                        for j in range(i + 1, len(df)):
                            if j in restricted_indices: continue
                            if df.loc[j, 'High'] >= target_activation_price:
                                sell_date = df.loc[j, 'Date']
                                sell_price = target_activation_price
                                status = "Closed"
                                break
                    else:
                        # Trailing Stop
                        trailing_active = False
                        peak_price = 0.0
                        
                        for j in range(i + 1, len(df)):
                            if j in restricted_indices: continue
                            day_high = df.loc[j, 'High']
                            day_low = df.loc[j, 'Low']
                            
                            if not trailing_active:
                                if day_high >= target_activation_price:
                                    trailing_active = True
                                    peak_price = day_high
                                    current_stop = peak_price * (1 - (trailing_stop_pct / 100.0))
                                    if day_low <= current_stop:
                                        sell_date = df.loc[j, 'Date']
                                        sell_price = current_stop
                                        status = "Closed"
                                        break
                            else:
                                current_stop = peak_price * (1 - (trailing_stop_pct / 100.0))
                                if day_low <= current_stop:
                                    sell_date = df.loc[j, 'Date']
                                    sell_price = current_stop
                                    status = "Closed"
                                    break
                                if day_high > peak_price:
                                    peak_price = day_high
                        
                trade_obj = {
                    "trade_id": len(potential_trades),
                    "buy_date": daily_date,
                    "buy_price": buy_price,
                    "drop_pct": f"{round(current_drop_pct, 2)}%",
                    "sell_date": sell_date,
                    "sell_price": sell_price,
                    "status": status
                }
                potential_trades.append(trade_obj)
                trades_by_date[daily_date]['buys'].append(trade_obj)
                if status == "Closed":
                    if sell_date not in trades
