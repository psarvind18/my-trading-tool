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
        
        # --- STRATEGY LOGIC ---
        buy_price = 0.0
        
        # 1. SWING / DIP LOGIC (Original)
        if strategy_mode in ["Swing Trading", "Dip Accumulation"]:
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
                            for j in range(i + 1, len(df)):
                                if j in restricted_indices: continue
                                if df.loc[j, 'High'] >= target_activation_price:
                                    sell_date = df.loc[j, 'Date']
                                    sell_price = target_activation_price
                                    status = "Closed"
                                    break
                        else:
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
                                        calc_stop = peak_price * (1 - (trailing_stop_pct / 100.0))
                                        current_stop = max(calc_stop, target_activation_price)
                                        if day_low <= current_stop:
                                            sell_date = df.loc[j, 'Date']
                                            sell_price = current_stop
                                            status = "Closed"
                                            break
                                else:
                                    if day_high > peak_price: peak_price = day_high
                                    calc_stop = peak_price * (1 - (trailing_stop_pct / 100.0))
                                    current_stop = max(calc_stop, target_activation_price)
                                    if day_low <= current_stop:
                                        sell_date = df.loc[j, 'Date']
                                        sell_price = current_stop
                                        status = "Closed"
                                        break
                    
                    trade_obj = {
                        "trade_id": len(potential_trades), "buy_date": daily_date,
                        "buy_price": buy_price, "drop_pct": f"{round(current_drop_pct, 2)}%",
                        "sell_date": sell_date, "sell_price": sell_price,
                        "status": status, "quantity": 0.0
                    }
                    potential_trades.append(trade_obj)
                    trades_by_date[daily_date]['buys'].append(trade_obj)
                    if status == "Closed":
                        if sell_date not in trades_by_date: trades_by_date[sell_date] = {'buys': [], 'sells': []}
                        trades_by_date[sell_date]['sells'].append(trade_obj)
                    
                    drop_level += 1
                else:
                    break

        # 2. TREND-FILTERED DIP LOGIC
        elif strategy_mode == "Trend-Filtered Dip":
            if pd.notnull(df.loc[i-1, 'SMA_Trend']):
                is_healthy = True
                for k in range(1, confirmation_days + 1):
                    if (i-k) < 0 or pd.isnull(df.loc[i-k, 'SMA_Trend']) or df.loc[i-k, 'Close'] <= df.loc[i-k, 'SMA_Trend']:
                        is_healthy = False
                        break
                
                if is_healthy:
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
                            
                            for j in range(i + 1, len(df)):
                                if df.loc[j, 'Close'] < df.loc[j, 'SMA_Trend']:
                                    sell_date = df.loc[j, 'Date']
                                    sell_price = df.loc[j, 'Close']
                                    status = "Closed"
                                    break
                            
                            trade_obj = {
                                "trade_id": len(potential_trades), "buy_date": daily_date,
                                "buy_price": buy_price, "drop_pct": f"{round(current_drop_pct, 2)}%",
                                "sell_date": sell_date, "sell_price": sell_price,
                                "status": status, "quantity": 0.0
                            }
                            potential_trades.append(trade_obj)
                            trades_by_date[daily_date]['buys'].append(trade_obj)
                            if status == "Closed":
                                if sell_date not in trades_by_date: trades_by_date[sell_date] = {'buys': [], 'sells': []}
                                trades_by_date[sell_date]['sells'].append(trade_obj)
                            
                            drop_level += 1
                        else:
                            break

        # 3. SMA CROSSOVER LOGIC
        elif strategy_mode == "SMA Crossover":
            if pd.notnull(df.loc[i, 'SMA_S']) and pd.notnull(df.loc[i, 'SMA_L']):
                if df.loc[i-1, 'SMA_S'] < df.loc[i-1, 'SMA_L'] and df.loc[i, 'SMA_S'] > df.loc[i, 'SMA_L']:
                    buy_price = df.loc[i, 'Close']
                    sell_date = pd.NaT
                    sell_price = 0.0
                    status = "Open"
                    for j in range(i + 1, len(df)):
                        if pd.notnull(df.loc[j, 'SMA_S']) and pd.notnull(df.loc[j, 'SMA_L']):
                            if df.loc[j-1, 'SMA_S'] > df.loc[j-1, 'SMA_L'] and df.loc[j, 'SMA_S'] < df.loc[j, 'SMA_L']:
                                sell_date = df.loc[j, 'Date']
                                sell_price = df.loc[j, 'Close']
                                status = "Closed"
                                break
                    trade_obj = {
                        "trade_id": len(potential_trades), "buy_date": daily_date,
                        "buy_price": buy_price, "drop_pct": "SMA Cross",
                        "sell_date": sell_date, "sell_price": sell_price,
                        "status": status, "quantity": 0.0
                    }
                    potential_trades.append(trade_obj)
                    trades_by_date[daily_date]['buys'].append(trade_obj)
                    if status == "Closed":
                        if sell_date not in trades_by_date: trades_by_date[sell_date] = {'buys': [], 'sells': []}
                        trades_by_date[sell_date]['sells'].append(trade_obj)

        # 4. RSI MEAN REVERSION LOGIC
        elif strategy_mode == "RSI Mean Reversion":
            if pd.notnull(df.loc[i, 'RSI']):
                if df.loc[i, 'RSI'] < rsi_buy:
                    buy_price = df.loc[i, 'Close']
                    sell_date = pd.NaT
                    sell_price = 0.0
                    status = "Open"
                    for j in range(i + 1, len(df)):
                        if pd.notnull(df.loc[j, 'RSI']):
                            if df.loc[j, 'RSI'] > rsi_sell:
                                sell_date = df.loc[j, 'Date']
                                sell_price = df.loc[j, 'Close']
                                status = "Closed"
                                break
                    trade_obj = {
                        "trade_id": len(potential_trades), "buy_date": daily_date,
                        "buy_price": buy_price, "drop_pct": f"RSI {round(df.loc[i, 'RSI'],1)}",
                        "sell_date": sell_date, "sell_price": sell_price,
                        "status": status, "quantity": 0.0
                    }
                    potential_trades.append(trade_obj)
                    trades_by_date[daily_date]['buys'].append(trade_obj)
                    if status == "Closed":
                        if sell_date not in trades_by_date: trades_by_date[sell_date] = {'buys': [], 'sells': []}
                        trades_by_date[sell_date]['sells'].append(trade_obj)

    # --- 2. Simulation (Shared Logic) ---
    wallet = initial_investment
    active_holdings = set() # trade_ids
    trade_decisions = {}
    trade_cash_flows = [] 
    portfolio_cash_flows = [ (df.iloc[0]['Date'], -initial_investment) ]
    
    total_interest_earned = 0.0
    total_dividends_earned = 0.0
    total_invested_capital = initial_investment
    
    executed_count = 0
    missed_count = 0
    current_total_shares = 0.0
    daily_history = []
    
    # B&H Benchmarking
    bh_shares = 0.0
    bh_wallet = initial_investment
    bh_cash_flows = [ (df.iloc[0]['Date'], -initial_investment) ]
    first_open_price = df.iloc[0]['Open'] 
    
    start_shares = bh_wallet / first_open_price
    bh_shares += start_shares
    bh_wallet = 0.0

    prev_sim_date = df.iloc[0]['Date']
    last_month_processed = -1
    
    for i in range(len(df)):
        curr_date = df.iloc[i]['Date']
        curr_close = df.iloc[i]['Close']
        
        if curr_date.month != last_month_processed:
            if monthly_investment > 0 and i > 0:
                wallet += monthly_investment
                portfolio_cash_flows.append((curr_date, -
