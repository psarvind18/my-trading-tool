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

def calculate_indicators(df):
    """Calculates SMA and RSI for strategy logic."""
    data = df.copy()
    
    # Simple Moving Averages
    data['SMA_Short'] = data['Close'].rolling(window=50).mean() # Default placeholders, dynamic in loop
    data['SMA_Long'] = data['Close'].rolling(window=200).mean()
    
    # RSI Calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

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

    # Unpack parameters
    strategy_mode = params['strategy_mode']
    
    # Strategy Specific Params
    buy_drop_pct = params.get('buy_drop_pct', 0)
    sell_profit_pct = params.get('sell_profit_pct', 0)
    use_trailing_stop = params.get('use_trailing_stop', False)
    trailing_stop_pct = params.get('trailing_stop_pct', 0)
    
    rsi_buy = params.get('rsi_buy', 30)
    rsi_sell = params.get('rsi_sell', 70)
    
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
    
    # State tracking for signal strategies
    active_position = False 
    
    for i in range(1, len(df)):
        daily_date = df.loc[i, 'Date']
        if daily_date not in trades_by_date:
            trades_by_date[daily_date] = {'buys': [], 'sells': []}
        
        if i in restricted_indices: continue 
        
        # --- STRATEGY LOGIC ---
        should_buy = False
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
                    # Logic for Sell Date calculation
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

        # 2. SMA CROSSOVER LOGIC
        elif strategy_mode == "SMA Crossover":
            # Check if we have data
            if pd.notnull(df.loc[i, 'SMA_S']) and pd.notnull(df.loc[i, 'SMA_L']):
                # Crossover: Short crosses ABOVE Long (Buy)
                if df.loc[i-1, 'SMA_S'] < df.loc[i-1, 'SMA_L'] and df.loc[i, 'SMA_S'] > df.loc[i, 'SMA_L']:
                    buy_price = df.loc[i, 'Close'] # Buy at Close
                    
                    # Find Sell Signal (Cross UNDER)
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

        # 3. RSI MEAN REVERSION LOGIC
        elif strategy_mode == "RSI Mean Reversion":
            if pd.notnull(df.loc[i, 'RSI']):
                # Buy if RSI < Buy Threshold
                if df.loc[i, 'RSI'] < rsi_buy:
                    buy_price = df.loc[i, 'Close']
                    
                    # Find Sell Signal (RSI > Sell Threshold)
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
    
    # Daily Tracking for Plotting
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
        
        # Monthly Contribution
        if curr_date.month != last_month_processed:
            if monthly_investment > 0 and i > 0:
                wallet += monthly_investment
                portfolio_cash_flows.append((curr_date, -monthly_investment))
                total_invested_capital += monthly_investment
                
                bh_wallet += monthly_investment
                bh_cash_flows.append((curr_date, -monthly_investment))
                new_shares = bh_wallet / curr_close
                bh_shares += new_shares
                bh_wallet = 0.0
            last_month_processed = curr_date.month

        # Interest
        days_delta = (curr_date - prev_sim_date).days
        if days_delta > 0:
            if wallet > 0:
                interest = wallet * (interest_rate_pct / 100.0 / 365.0) * days_delta
                wallet += interest
                total_interest_earned += interest
            if bh_wallet > 0:
                bh_interest = bh_wallet * (interest_rate_pct / 100.0 / 365.0) * days_delta
                bh_wallet += bh_interest

        # Dividends
        if enable_dividends:
            today_div_amount = df.loc[i, 'Dividends']
            if today_div_amount > 0:
                if current_total_shares > 0:
                    payout = current_total_shares * today_div_amount
                    wallet += payout
                    total_dividends_earned += payout
                if bh_shares > 0:
                    bh_payout = bh_shares * today_div_amount
                    bh_wallet += bh_payout

        # Trading
        if curr_date in trades_by_date:
            day_activity = trades_by_date[curr_date]
            
            # Sells
            for t in day_activity['sells']:
                if t['trade_id'] in active_holdings:
                    qty_held = t['quantity']
                    revenue = t['sell_price'] * qty_held
                    wallet += revenue
                    active_holdings.remove(t['trade_id'])
                    current_total_shares -= qty_held
                    trade_cash_flows.append((curr_date, revenue))
            
            # Buys
            for t in day_activity['buys']:
                # Filter logic: For SMA/RSI, we might get multiple buy signals.
                # If we are already holding this exact "trade idea", we skip to avoid infinite buying on same day
                if t['trade_id'] in trade_decisions: continue

                qty_to_buy = 0.0
                cost = 0.0
                
                if trade_size_type == "Fixed Shares":
                    qty_to_buy = float(shares_per_trade)
                    cost = qty_to_buy * t['buy_price']
                    
                    if wallet >= cost:
                        wallet -= cost
                        t['quantity'] = qty_to_buy
                        active_holdings.add(t['trade_id'])
                        current_total_shares += qty_to_buy
                        trade_decisions[t['trade_id']] = "Executed"
                        executed_count += 1
                        trade_cash_flows.append((curr_date, -cost))
                    else:
                        trade_decisions[t['trade_id']] = "Missed"
                        missed_count += 1
                else: 
                    target_spend = max_trade_amt
                    possible_spend = min(wallet, target_spend)
                    
                    if possible_spend >= min_trade_amt:
                        cost = possible_spend
                        qty_to_buy = cost / t['buy_price']
                        wallet -= cost
                        t['quantity'] = qty_to_buy
                        active_holdings.add(t['trade_id'])
                        current_total_shares += qty_to_buy
                        trade_decisions[t['trade_id']] = "Executed"
                        executed_count += 1
                        trade_cash_flows.append((curr_date, -cost))
                    else:
                        trade_decisions[t['trade_id']] = "Missed"
                        missed_count += 1
        
        daily_open_value = current_total_shares * curr_close
        daily_total_value = wallet + daily_open_value
        daily_bh_value = (bh_shares * curr_close) + bh_wallet
        
        daily_history.append({
            "Date": curr_date,
            "Total Value": daily_total_value,
            "Cash": wallet,
            "Open Positions": daily_open_value,
            "Buy & Hold": daily_bh_value
        })
        
        prev_sim_date = curr_date

    # --- 4. Valuation ---
    last_close_price = df.iloc[-1]['Close']
    final_date = df.iloc[-1]['Date']
    
    open_position_value = 0
    for t in potential_trades:
        if trade_decisions.get(t['trade_id']) == "Executed" and t['status'] == "Open":
            val = last_close_price * t['quantity']
            open_position_value += val
            trade_cash_flows.append((final_date, val))

    final_strategy_value = wallet + open_position_value
    portfolio_cash_flows.append((final_date, final_strategy_value))
    
    final_bh_value = (bh_shares * last_close_price) + bh_wallet
    bh_cash_flows.append((final_date, final_bh_value))

    # --- Metrics ---
    strategy_xirr = xirr(portfolio_cash_flows)
    bh_xirr = xirr(bh_cash_flows)
    trade_efficiency_xirr = xirr(trade_cash_flows)
    
    return {
        "final_value": final_strategy_value,
        "invested_capital": total_invested_capital,
        "strategy_xirr": strategy_xirr,
        "bh_xirr": bh_xirr,
        "trade_xirr": trade_efficiency_xirr,
        "executed_trades": executed_count,
        "missed_trades": missed_count,
        "passive_income": total_interest_earned + total_dividends_earned,
        "wallet_cash": wallet,
        "open_value": open_position_value,
        "trades": potential_trades,
        "decisions": trade_decisions,
        "dividend_events": dividend_events,
        "bh_final_value": final_bh_value,
        "daily_history": daily_history
    }

# --- Main App ---
st.title("🛡️ Algorithmic Strategy Optimizer")

if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None
if 'stock_info' not in st.session_state:
    st.session_state['stock_info'] = {}
    
# Initialize Results in Session State to fix disappearance
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Loading")
    ticker_symbol = st.text_input("Ticker Symbol", value="VOO").upper()
    start_input = st.date_input("Start Date", value=date.today().replace(year=date.today().year - 2))
    end_input = st.date_input("End Date", value=date.today())
    
    if st.button("Step 1: Fetch Data"):
        with st.spinner(f"Downloading {ticker_symbol}..."):
            df, info = fetch_data_from_yahoo(ticker_symbol, start_input, end_input)
            if df is not None and not df.empty:
                st.session_state['stock_data'] = df
                st.session_state['stock_info'] = info
                st.session_state['data_ticker'] = ticker_symbol
                st.success(f"Loaded {len(df)} days!")
            else:
                st.error(f"Error: {info}")

    st.divider()
    
    st.header("2. Strategy Settings")
    # UPDATED DROPDOWN
    strategy_mode = st.selectbox("Strategy Type", ["Swing Trading", "Dip Accumulation", "SMA Crossover", "RSI Mean Reversion"])
    
    # Initialize vars
    buy_drop_pct = 0.0
    sell_profit_pct = 0.0
    use_trailing_stop = False
    trailing_stop_pct = 0.0
    sma_short = 50
    sma_long = 200
    rsi_buy = 30
    rsi_sell = 70

    if strategy_mode in ["Swing Trading", "Dip Accumulation"]:
        buy_drop_pct = st.number_input("Buy Drop Step (%)", value=1.0, step=0.1)
        if strategy_mode == "Swing Trading":
            sell_profit_pct = st.number_input("Activation Target (%)", value=4.0, step=0.1)
            use_trailing_stop = st.checkbox("Enable Trailing Stop", value=True)
            if use_trailing_stop:
                trailing_stop_pct = st.number_input("Trailing Stop (%) (Base Value)", value=2.0, step=0.1)
    
    elif strategy_mode == "SMA Crossover":
        st.info("Buys when Short SMA crosses ABOVE Long SMA. Sells when Short crosses BELOW.")
        sma_short = st.number_input("Short SMA (Days)", value=50, step=5)
        sma_long = st.number_input("Long SMA (Days)", value=200, step=5)
        

    elif strategy_mode == "RSI Mean Reversion":
        st.info("Buys when RSI < Buy Level. Sells when RSI > Sell Level.")
        rsi_buy = st.number_input("RSI Buy Level (Oversold)", value=30, step=5)
        rsi_sell = st.number_input("RSI Sell Level (Overbought)", value=70, step=5)
        

    st.divider()
    st.header("3. Financials")
    interest_rate_pct = st.number_input("Cash Interest (%)", value=3.75, step=0.25, format="%.2f")
    enable_dividends = st.checkbox("Include Dividends", True)
    restrict_ex_date = st.checkbox("Restrict Ex-Date", True) if enable_dividends else False
    
    st.divider()
    st.header("4. Wallet & Sizing")
    currency_symbol = st.text_input("Currency", "$")
    initial_investment = st.number_input("Initial Inv.", value=1000.0, step=500.0)
    monthly_investment = st.number_input("Monthly Contrib.", value=0.0, step=100.0)
    
    trade_size_type = st.selectbox("Trade Size Type", ["Fixed Shares", "Dollar Amount"])
    shares_per_trade = 1
    min_trade_amt = 0.0
    max_trade_amt = 0.0
    
    if trade_size_type == "Fixed Shares":
        shares_per_trade = st.number_input("Shares per Trade", value=1, step=1)
    else:
        c1, c2 = st.columns(2)
        min_trade_amt = c1.number_input("Min Trade $", value=100.0, step=50.0)
        max_trade_amt = c2.number_input("Max Trade $", value=1000.0, step=50.0)

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Single Backtest", "🚀 Optimizer (Parameter Sweep)"])

if st.session_state['stock_data'] is not None:
    df = st.session_state['stock_data']
    info = st.session_state['stock_info']
    
    # Pack parameters
    current_params = {
        'strategy_mode': strategy_mode,
        'buy_drop_pct': buy_drop_pct,
        'sell_profit_pct': sell_profit_pct,
        'use_trailing_stop': use_trailing_stop,
        'trailing_stop_pct': trailing_stop_pct,
        'sma_short': sma_short,
        'sma_long': sma_long,
        'rsi_buy': rsi_buy,
        'rsi_sell': rsi_sell,
        'interest_rate_pct': interest_rate_pct,
        'enable_dividends': enable_dividends,
        'restrict_ex_date': restrict_ex_date,
        'initial_investment': initial_investment,
        'monthly_investment': monthly_investment,
        'trade_size_type': trade_size_type,
        'shares_per_trade': shares_per_trade,
        'min_trade_amt': min_trade_amt,
        'max_trade_amt': max_trade_amt
    }

    # --- TAB 1: SINGLE RUN ---
    with tab1:
        if st.button("Run Single Backtest"):
            res = run_simulation(df, current_params)
            st.session_state['sim_results'] = res
        
        if st.session_state['sim_results'] is not None:
            res = st.session_state['sim_results']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Value", f"{currency_symbol}{res['final_value']:,.2f}", delta=f"Inv: {currency_symbol}{res['invested_capital']:,.0f}")
            c2.metric("Strategy XIRR", f"{res['strategy_xirr']:.2%}")
            c3.metric("Buy & Hold XIRR", f"{res['bh_xirr']:.2%}")
            c4.metric("Trade Efficiency", f"{res['trade_xirr']:.2%}")
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Trades", f"{res['executed_trades']} / {res['executed_trades'] + res['missed_trades']}")
            c6.metric("Passive Income", f"{currency_symbol}{res['passive_income']:,.2f}")
            c7.metric("Cash Balance", f"{currency_symbol}{res['wallet_cash']:,.2f}")
            c8.metric("Open Value", f"{currency_symbol}{res['open_value']:,.2f}")

            if enable_dividends and res['dividend_events']:
                with st.expander(f"📅 Dividend Schedule ({len(res['dividend_events'])})"):
                    st.dataframe(pd.DataFrame(res['dividend_events']))
            
            st.subheader("Trade Log")
            logs = []
            last_close = df.iloc[-1]['Close']
            for t in res['trades']:
                decision = res['decisions'].get(t['trade_id'], "Missed")
                p_share = 0.0
                s_price = 0.0
                qty = t['quantity']
                
                if decision == "Executed":
                    s_price = t['sell_price'] if t['status'] == "Closed" else last_close
                    p_share = s_price - t['buy_price']
                
                logs.append({
                    "Date": t['buy_date'].strftime('%Y-%m-%d'),
                    "Buy": f"{currency_symbol}{t['buy_price']:.2f}",
                    "Sell": f"{currency_symbol}{s_price:.2f}" if decision == "Executed" else "-",
                    "Qty": f"{qty:.4f}" if decision == "Executed" else "-",
                    "Profit": f"{currency_symbol}{p_share * qty:.2f}" if decision == "Executed" else "-",
                    "Status": t['status'],
                    "Drop/Signal": t['drop_pct']
                })
            st.dataframe(pd.DataFrame(logs), use_container_width=True)
            
            st.subheader("📈 Portfolio Growth Over Time")
            hist_df = pd.DataFrame(res['daily_history'])
            all_metrics = ['Total Value', 'Buy & Hold', 'Cash', 'Open Positions']
            chart_df = hist_df.melt(id_vars='Date', value_vars=all_metrics, 
                                    var_name='Metric', value_name='Value')
            
            selected_metrics = st.multiselect("Select Metrics:", options=all_metrics, default=all_metrics)
            
            if selected_metrics:
                filtered_chart_df = chart_df[chart_df['Metric'].isin(selected_metrics)]
                chart = alt.Chart(filtered_chart_df).mark_line().encode(
                    x='Date:T',
                    y=alt.Y('Value:Q', title=f'Value ({currency_symbol})'),
                    color='Metric:N',
                    tooltip=['Date', 'Metric', 'Value']
                ).properties(height=400).interactive()
                st.altair_chart(chart, use_container_width=True)

    # --- TAB 2: OPTIMIZER ---
    with tab2:
        st.write("Automatically test different parameters to find the 'Sweet Spot'.")
        
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            optimize_target = st.selectbox("Parameter to Optimize", 
                                         ["Trailing Stop %", "Activation Target %", "Buy Drop %", "SMA Short", "RSI Buy"])
        with opt_col2:
            st.write("Range Settings")
            r_start = st.number_input("Start", value=1.0, step=0.5)
            r_end = st.number_input("End", value=10.0, step=0.5)
            r_step = st.number_input("Step", value=0.5, step=0.1)

        if st.button("Run Optimization Sweep"):
            if strategy_mode == "SMA Crossover" and optimize_target not in ["SMA Short"]:
                 st.error("For SMA Crossover, you can only optimize 'SMA Short' in this demo.")
            elif strategy_mode == "RSI Mean Reversion" and optimize_target not in ["RSI Buy"]:
                 st.error("For RSI, you can only optimize 'RSI Buy' in this demo.")
            elif strategy_mode in ["Swing Trading", "Dip Accumulation"] and optimize_target in ["SMA Short", "RSI Buy"]:
                 st.error("Select a relevant parameter for Swing Trading.")
            else:
                results_sweep = []
                import numpy as np
                
                test_values = np.arange(r_start, r_end + 0.001, r_step)
                bar = st.progress(0)
                best_xirr = -999.0
                best_val = 0.0
                
                for idx, val in enumerate(test_values):
                    temp_params = current_params.copy()
                    if optimize_target == "Trailing Stop %":
                        temp_params['use_trailing_stop'] = True
                        temp_params['trailing_stop_pct'] = val
                    elif optimize_target == "Activation Target %":
                        temp_params['sell_profit_pct'] = val
                    elif optimize_target == "Buy Drop %":
                        temp_params['buy_drop_pct'] = val
                    elif optimize_target == "SMA Short":
                        temp_params['sma_short'] = int(val)
                    elif optimize_target == "RSI Buy":
                        temp_params['rsi_buy'] = int(val)
                    
                    res = run_simulation(df, temp_params)
                    results_sweep.append({
                        "Parameter Value": round(val, 2),
                        "Strategy XIRR": res['strategy_xirr'] * 100.0,
                        "Profit": res['final_value'] - res['invested_capital']
                    })
                    
                    if res['strategy_xirr'] > best_xirr:
                        best_xirr = res['strategy_xirr']
                        best_val = val
                    bar.progress((idx + 1) / len(test_values))
                
                bar.empty()
                res_df = pd.DataFrame(results_sweep)
                
                st.success(f"🏆 Best {optimize_target}: **{best_val:.2f}** (XIRR: {best_xirr:.2%})")
                
                chart = alt.Chart(res_df).mark_line(point=True).encode(
                    x=alt.X('Parameter Value', title=f'{optimize_target}'),
                    y=alt.Y('Strategy XIRR', title='Return (XIRR %)'),
                    tooltip=['Parameter Value', 'Strategy XIRR', 'Profit']
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(res_df.set_index("Parameter Value"))

else:
    st.info("👈 Step 1: Load Data to begin.")
