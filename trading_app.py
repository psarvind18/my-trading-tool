import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import numpy as np
from datetime import date
import traceback

# --- Configuration ---
st.set_page_config(page_title="Algorithmic Strategy Optimizer", layout="wide")

# Wrap the entire app in a try-except to prevent silent "Blank Page" crashes
try:
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
        df = df_raw.copy()
        
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

        strategy_mode = params['strategy_mode']
        buy_drop_pct = params.get('buy_drop_pct', 0.0)
        sell_profit_pct = params.get('sell_profit_pct', 0.0)
        use_trailing_stop = params.get('use_trailing_stop', False)
        trailing_stop_pct = params.get('trailing_stop_pct', 0.0)
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

        potential_trades = []
        trades_by_date = {} 
        
        for i in range(1, len(df)):
            daily_date = df.loc[i, 'Date']
            if daily_date not in trades_by_date:
                trades_by_date[daily_date] = {'buys': [], 'sells': []}
            
            if i in restricted_indices: continue 
            
            buy_price = 0.0
            
            # --- STRATEGY LOGIC ---
            if strategy_mode in ["Swing Trading", "Dip Accumulation"]:
                prev_close = df.loc[i-1, 'Close']
                daily_low = df.loc[i, 'Low']
                
                drop_level = 1
                while True:
                    if buy_drop_pct <= 0: break # Infinite loop safety
                        
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
                            if buy_drop_pct <= 0: break # Infinite loop safety
                                
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

        # --- 2. Simulation Execution ---
        wallet = initial_investment
        active_holdings = set()
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
                    portfolio_cash_flows.append((curr_date, -monthly_investment))
                    total_invested_capital += monthly_investment
                    
                    bh_wallet += monthly_investment
                    bh_cash_flows.append((curr_date, -monthly_investment))
                    new_shares = bh_wallet / curr_close
                    bh_shares += new_shares
                    bh_wallet = 0.0
                last_month_processed = curr_date.month

            days_delta = (curr_date - prev_sim_date).days
            if days_delta > 0:
                if wallet > 0:
                    interest = wallet * (interest_rate_pct / 100.0 / 365.0) * days_delta
                    wallet += interest
                    total_interest_earned += interest
                if bh_wallet > 0:
                    bh_interest = bh_wallet * (interest_rate_pct / 100.0 / 365.0) * days_delta
                    bh_wallet += bh_interest

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

            if curr_date in trades_by_date:
                day_activity = trades_by_date[curr_date]
                
                for t in day_activity['sells']:
                    if t['trade_id'] in active_holdings:
                        qty_held = t['quantity']
                        revenue = t['sell_price'] * qty_held
                        wallet += revenue
                        active_holdings.remove(t['trade_id'])
                        current_total_shares -= qty_held
                        trade_cash_flows.append((curr_date, revenue))
                
                for t in day_activity['buys']:
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
        
    if 'sim_results' not in st.session_state:
        st.session_state['sim_results'] = None
    if 'baseline_results' not in st.session_state:
        st.session_state['baseline_results'] = None

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
        strategy_mode = st.selectbox("Strategy Type", ["Trend-Filtered Dip", "Swing Trading", "Dip Accumulation", "SMA Crossover", "RSI Mean Reversion"])
        
        buy_drop_pct = 0.0
        sell_profit_pct = 0.0
        use_trailing_stop = False
        trailing_stop_pct = 0.0
        sma_short = 50
        sma_long = 200
        rsi_buy = 30
        rsi_sell = 70
        trend_sma = 200
        confirmation_days = 3

        if strategy_mode in ["Swing Trading", "Dip Accumulation", "Trend-Filtered Dip"]:
            buy_drop_pct = st.number_input("Buy Drop Step (%)", value=1.0, step=0.1)
            
            if strategy_mode == "Trend-Filtered Dip":
                st.info("Buys dips ONLY when price is in a confirmed uptrend. Sells everything when trend breaks.")
                trend_sma = st.number_input("Trend Filter (SMA Days)", value=200, step=10)
                confirmation_days = st.number_input("Confirmation (Days above SMA)", value=3, step=1)
                
            elif strategy_mode == "Swing Trading":
                sell_profit_pct = st.number_input("Activation Target (%)", value=4.0, step=0.1)
                use_trailing_stop = st.checkbox("Enable Trailing Stop", value=True)
                if use_trailing_stop:
                    trailing_stop_pct = st.number_input("Trailing Stop (%) (Base Value)", value=2.0, step=0.1)
        
        elif strategy_mode == "SMA Crossover":
            st.info("Buys when Short SMA crosses ABOVE Long SMA. Sells when Short crosses BELOW.")
            sma_short = st.number_input("Short SMA (Days)", value=50, step=5)
            sma_long = st.number_input("Long SMA (Days)", value=200
