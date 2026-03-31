import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import numpy as np
from datetime import date
import traceback

# Unlock Altair's row limit so lines don't disappear on multi-year backtests
alt.data_transformers.disable_max_rows()

# --- Configuration ---
st.set_page_config(page_title="Algorithmic Strategy Optimizer", layout="wide")

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
        
        # --- Pre-calculate Indicators ---
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
        elif params['strategy_mode'] == "Bollinger Bands":
            df['BB_SMA'] = df['Close'].rolling(window=params['bb_window']).mean()
            df['BB_STD'] = df['Close'].rolling(window=params['bb_window']).std()
            df['BB_Upper'] = df['BB_SMA'] + (params['bb_std'] * df['BB_STD'])
            df['BB_Lower'] = df['BB_SMA'] - (params['bb_std'] * df['BB_STD'])

        strategy_mode = params['strategy_mode']
        buy_drop_pct = params.get('buy_drop_pct', 0.0)
        sell_profit_pct = params.get('sell_profit_pct', 0.0)
        use_trailing_stop = params.get('use_trailing_stop', False)
        trailing_stop_pct = params.get('trailing_stop_pct', 0.0)
        
        rsi_buy = params.get('rsi_buy', 30)
        rsi_sell = params.get('rsi_sell', 70)
        trend_sma = params.get('trend_sma', 200)
        confirmation_days = params.get('confirmation_days', 3)
        bb_window = params.get('bb_window', 50)
        bb_std = params.get('bb_std', 2.0)
        
        interest_rate_pct = params['interest_rate_pct']
        enable_dividends = params['enable_dividends']
        restrict_ex_date = params['restrict_ex_date']
        
        enable_taxes = params.get('enable_taxes', False)
        st_tax_rate = params.get('st_tax_rate', 0.25)
        lt_tax_rate = params.get('lt_tax_rate', 0.15)
        
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
        
        # --- 1. GENERATE DAILY SIGNALS ---
        for i in range(1, len(df)):
            daily_date = df.loc[i, 'Date']
            if daily_date not in trades_by_date:
                trades_by_date[daily_date] = {'buys': [], 'sells': []}
            
            if i in restricted_indices: continue 
            
            buy_price = 0.0
            
            if strategy_mode in ["Swing Trading", "Dip Accumulation"]:
                prev_close = df.loc[i-1, 'Close']
                daily_low = df.loc[i, 'Low']
                
                drop_level = 1
                while True:
                    if buy_drop_pct <= 0: break
                        
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
                            if buy_drop_pct <= 0: break
                                
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

            elif strategy_mode == "Bollinger Bands":
                if pd.notnull(df.loc[i-1, 'BB_Lower']):
                    # Buy when price drops below yesterday's Lower Band
                    if df.loc[i, 'Low'] <= df.loc[i-1, 'BB_Lower']:
                        buy_price = df.loc[i-1, 'BB_Lower']
                        sell_date = pd.NaT
                        sell_price = 0.0
                        status = "Open"
                        for j in range(i + 1, len(df)):
                            if pd.notnull(df.loc[j-1, 'BB_Upper']):
                                # Sell when price jumps above yesterday's Upper Band
                                if df.loc[j, 'High'] >= df.loc[j-1, 'BB_Upper']:
                                    sell_date = df.loc[j, 'Date']
                                    sell_price = df.loc[j-1, 'BB_Upper']
                                    status = "Closed"
                                    break
                        trade_obj = {
                            "trade_id": len(potential_trades), "buy_date": daily_date,
                            "buy_price": buy_price, "drop_pct": "BB Touch",
                            "sell_date": sell_date, "sell_price": sell_price,
                            "status": status, "quantity": 0.0
                        }
                        potential_trades.append(trade_obj)
                        trades_by_date[daily_date]['buys'].append(trade_obj)
                        if status == "Closed":
                            if sell_date not in trades_by_date: trades_by_date[sell_date] = {'buys': [], 'sells': []}
                            trades_by_date[sell_date]['sells'].append(trade_obj)


        # --- 2. SIMULATION EXECUTION ---
        wallet = initial_investment
        active_holdings = set()
        trade_decisions = {}
        trade_cash_flows = [] 
        portfolio_cash_flows = [ (df.iloc[0]['Date'], -initial_investment) ]
        
        yearly_realized_gains = {} 
        taxes_paid_years = set()
        total_taxes_paid = 0.0
        tax_events_log = []
        
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
        months_passed = 0
        
        # Value Averaging Initialization (Day 1)
        if strategy_mode == "Value Averaging":
            qty = initial_investment / first_open_price
            wallet -= initial_investment
            current_total_shares += qty
            t_obj = {
                "trade_id": 0, "buy_date": df.iloc[0]['Date'],
                "buy_price": first_open_price, "drop_pct": "VA Init",
                "sell_date": pd.NaT, "sell_price": 0.0,
                "status": "Open", "quantity": qty
            }
            potential_trades.append(t_obj)
            trade_decisions[0] = "Executed"
            active_holdings.add(0)
            executed_count += 1
            # Note: portfolio_cash_flows already logged -initial_investment
        
        for i in range(len(df)):
            curr_date = df.iloc[i]['Date']
            curr_close = df.iloc[i]['Close']
            
            # --- TAX PAYMENT LOGIC (April 1st) ---
            if enable_taxes and curr_date.month >= 4:
                prev_year = curr_date.year - 1
                if prev_year not in taxes_paid_years:
                    taxes_paid_years.add(prev_year)
                    if prev_year in yearly_realized_gains:
                        st_gains = yearly_realized_gains[prev_year]['st']
                        lt_gains = yearly_realized_gains[prev_year]['lt']
                        
                        tax_bill = 0.0
                        if st_gains >= 0 and lt_gains >= 0:
                            tax_bill = (st_gains * st_tax_rate) + (lt_gains * lt_tax_rate)
                        elif st_gains < 0 and lt_gains > 0:
                            net = st_gains + lt_gains
                            if net > 0: tax_bill = net * lt_tax_rate
                        elif lt_gains < 0 and st_gains > 0:
                            net = st_gains + lt_gains
                            if net > 0: tax_bill = net * st_tax_rate
                            
                        if tax_bill > 0:
                            if wallet >= tax_bill:
                                wallet -= tax_bill
                            else:
                                shortfall = tax_bill - wallet
                                for t_id in list(active_holdings):
                                    t_obj = potential_trades[t_id]
                                    qty = t_obj['quantity']
                                    revenue = qty * curr_close
                                    
                                    wallet += revenue
                                    shortfall -= revenue
                                    active_holdings.remove(t_id)
                                    current_total_shares -= qty
                                    
                                    f_profit = revenue - (t_obj['buy_price'] * qty)
                                    f_days = (curr_date - t_obj['buy_date']).days
                                    c_year = curr_date.year
                                    if c_year not in yearly_realized_gains:
                                        yearly_realized_gains[c_year] = {'st': 0.0, 'lt': 0.0}
                                    if f_days > 365:
                                        yearly_realized_gains[c_year]['lt'] += f_profit
                                    else:
                                        yearly_realized_gains[c_year]['st'] += f_profit
                                        
                                    t_obj['status'] = "Tax Liquidation"
                                    t_obj['sell_date'] = curr_date
                                    t_obj['sell_price'] = curr_close
                                    
                                    if shortfall <= 0: break
                                
                                wallet -= tax_bill
                            
                            total_taxes_paid += tax_bill
                            tax_events_log.append({"Date": curr_date, "Year Taxed": prev_year, "Amount": tax_bill})
            
            # --- MONTHLY ACTIONS ---
            if curr_date.month != last_month_processed:
                if i > 0: months_passed += 1
                
                # Standard Contribution
                if monthly_investment > 0 and i > 0:
                    wallet += monthly_investment
                    portfolio_cash_flows.append((curr_date, -monthly_investment))
                    total_invested_capital += monthly_investment
                    
                    bh_wallet += monthly_investment
                    bh_cash_flows.append((curr_date, -monthly_investment))
                    new_shares = bh_wallet / curr_close
                    bh_shares += new_shares
                    bh_wallet = 0.0
                
                # Value Averaging Specific Monthly Rebalance
                if strategy_mode == "Value Averaging" and i > 0:
                    target_value = initial_investment + (months_passed * monthly_investment)
                    current_holdings_val = current_total_shares * curr_close
                    diff = target_value - current_holdings_val
                    
                    if diff > 0:
                        # Buy Shares to catch up
                        spend = min(diff, wallet)
                        if spend > 0:
                            qty = spend / curr_close
                            wallet -= spend
                            current_total_shares += qty
                            t_obj = {
                                "trade_id": len(potential_trades), "buy_date": curr_date,
                                "buy_price": curr_close, "drop_pct": "VA Buy",
                                "sell_date": pd.NaT, "sell_price": 0.0,
                                "status": "Open", "quantity": qty
                            }
                            potential_trades.append(t_obj)
                            trade_decisions[t_obj['trade_id']] = "Executed"
                            active_holdings.add(t_obj['trade_id'])
                            executed_count += 1
                            trade_cash_flows.append((curr_date, -spend))
                    elif diff < 0:
                        # Sell Shares to lock in excess profit
                        target_revenue = min(abs(diff), current_holdings_val)
                        revenue_collected = 0.0
                        
                        sorted_holdings = sorted(list(active_holdings))
                        for t_id in sorted_holdings:
                            if revenue_collected >= target_revenue: break
                            t = potential_trades[t_id]
                            pos_value = t['quantity'] * curr_close
                            
                            if pos_value <= (target_revenue - revenue_collected):
                                # Liquidate entire lot
                                wallet += pos_value
                                revenue_collected += pos_value
                                current_total_shares -= t['quantity']
                                active_holdings.remove(t_id)
                                t['status'] = "VA Sell"
                                t['sell_date'] = curr_date
                                t['sell_price'] = curr_close
                                trade_cash_flows.append((curr_date, pos_value))
                                
                                if enable_taxes:
                                    profit = pos_value - (t['buy_price'] * t['quantity'])
                                    days_held = (curr_date - t['buy_date']).days
                                    c_year = curr_date.year
                                    if c_year not in yearly_realized_gains: yearly_realized_gains[c_year] = {'st': 0.0, 'lt': 0.0}
                                    if days_held > 365: yearly_realized_gains[c_year]['lt'] += profit
                                    else: yearly_realized_gains[c_year]['st'] += profit
                            else:
                                # Liquidate partial lot
                                needed_revenue = target_revenue - revenue_collected
                                qty_to_sell = needed_revenue / curr_close
                                
                                wallet += needed_revenue
                                revenue_collected += needed_revenue
                                current_total_shares -= qty_to_sell
                                t['quantity'] -= qty_to_sell
                                
                                # Log the partial sell for tracking
                                sold_obj = {
                                    "trade_id": len(potential_trades), "buy_date": t['buy_date'],
                                    "buy_price": t['buy_price'], "drop_pct": "VA Sell",
                                    "sell_date": curr_date, "sell_price": curr_close,
                                    "status": "VA Sell", "quantity": qty_to_sell
                                }
                                potential_trades.append(sold_obj)
                                trade_decisions[sold_obj['trade_id']] = "Executed"
                                trade_cash_flows.append((curr_date, needed_revenue))
                                
                                if enable_taxes:
                                    profit = needed_revenue - (t['buy_price'] * qty_to_sell)
                                    days_held = (curr_date - t['buy_date']).days
                                    c_year = curr_date.year
                                    if c_year not in yearly_realized_gains: yearly_realized_gains[c_year] = {'st': 0.0, 'lt': 0.0}
                                    if days_held > 365: yearly_realized_gains[c_year]['lt'] += profit
                                    else: yearly_realized_gains[c_year]['st'] += profit
                                break
                            
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

            # Trading Execution (For non-VA signal-based strategies)
            if strategy_mode != "Value Averaging" and curr_date in trades_by_date:
                day_activity = trades_by_date[curr_date]
                
                for t in day_activity['sells']:
                    if t['trade_id'] in active_holdings:
                        qty_held = t['quantity']
                        revenue = t['sell_price'] * qty_held
                        wallet += revenue
                        active_holdings.remove(t['trade_id'])
                        current_total_shares -= qty_held
                        trade_cash_flows.append((curr_date, revenue))
                        
                        if enable_taxes:
                            profit = revenue - (t['buy_price'] * qty_held)
                            days_held = (curr_date - t['buy_date']).days
                            c_year = curr_date.year
                            if c_year not in yearly_realized_gains: yearly_realized_gains[c_year] = {'st': 0.0, 'lt': 0.0}
                            if days_held > 365: yearly_realized_gains[c_year]['lt'] += profit
                            else: yearly_realized_gains[c_year]['st'] += profit
                
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

        # --- 4. Final Valuation ---
        last_close_price = df.iloc[-1]['Close']
        final_date = df.iloc[-1]['Date']
        
        open_position_value = 0
        for t in potential_trades:
            if trade_decisions.get(t['trade_id']) == "Executed" and t.get('status') == "Open":
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
            "total_taxes_paid": total_taxes_paid,
            "tax_events": tax_events_log,
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
        start_input = st.date_input("Start Date", value=date.today().replace(year=date.today().year - 2), min_value=date(1990, 1, 1))
        end_input = st.date_input("End Date", value=date.today(), min_value=date(1990, 1, 1))
        
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
        strategy_mode = st.selectbox("Strategy Type", [
            "Value Averaging", "Bollinger Bands", "Swing Trading", 
            "Trend-Filtered Dip", "Dip Accumulation", "SMA Crossover", "RSI Mean Reversion"
        ])
        
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
        bb_window = 50
        bb_std = 2.0

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
        
        elif strategy_mode == "Value Averaging":
            st.info("Dynamically buys and sells shares to force the portfolio value to increase by exactly your 'Monthly Contribution' amount every month.")

        elif strategy_mode == "Bollinger Bands":
            st.info("Buys when price touches the Lower Band. Sells when price touches the Upper Band.")
            bb_window = st.number_input("BB Window (Days)", value=50, step=5)
            bb_std = st.number_input("Standard Deviations", value=2.0, step=0.1)

        elif strategy_mode == "SMA Crossover":
            st.info("Buys when Short SMA crosses ABOVE Long SMA. Sells when Short crosses BELOW.")
            sma_short = st.number_input("Short SMA (Days)", value=50, step=5)
            sma_long = st.number_input("Long SMA (Days)", value=200, step=5)

        elif strategy_mode == "RSI Mean Reversion":
            st.info("Buys when RSI < Buy Level. Sells when RSI > Sell Level.")
            rsi_buy = st.number_input("RSI Buy Level (Oversold)", value=30, step=5)
            rsi_sell = st.number_input("RSI Sell Level (Overbought)", value=70, step=5)

        st.divider()
        st.header("3. Baseline Comparison")
        st.info("Dip Accumulation is always calculated as a baseline.")
        baseline_buy_drop_pct = st.number_input("Baseline Buy Drop (%)", value=1.0, step=0.1)

        st.divider()
        st.header("4. Financials & Taxes")
        interest_rate_pct = st.number_input("Cash Interest (%)", value=3.75, step=0.25, format="%.2f")
        enable_dividends = st.checkbox("Include Dividends", True)
        restrict_ex_date = st.checkbox("Restrict Ex-Date", True) if enable_dividends else False
        
        st.write("**Capital Gains Tax**")
        enable_taxes = st.checkbox("Enable Taxes (Paid April 1)", value=True)
        st_tax_rate = 0.25
        lt_tax_rate = 0.15
        if enable_taxes:
            c_t1, c_t2 = st.columns(2)
            st_tax_rate = c_t1.number_input("Short Term Rate", value=0.25, step=0.05)
            lt_tax_rate = c_t2.number_input("Long Term Rate", value=0.15, step=0.05)
            st.caption("Deducts taxes from cash balance. If insufficient, liquidates shares.")
        
        st.divider()
        st.header("5. Wallet & Sizing")
        currency_symbol = st.text_input("Currency", "$")
        initial_investment = st.number_input("Initial Inv.", value=1000.0, step=500.0)
        monthly_investment = st.number_input("Monthly Contrib.", value=500.0, step=100.0)
        
        trade_size_type = st.selectbox("Trade Size Type", ["Dollar Amount", "Fixed Shares"])
        shares_per_trade = 1
        min_trade_amt = 0.0
        max_trade_amt = 0.0
        
        if trade_size_type == "Fixed Shares":
            shares_per_trade = st.number_input("Shares per Trade", value=1, step=1)
        else:
            c1, c2 = st.columns(2)
            min_trade_amt = c1.number_input("Min Trade $", value=100.0, step=50.0)
            max_trade_amt = c2.number_input("Max Trade $", value=1000.0, step=50.0)

    tab1, tab2 = st.tabs(["📊 Single Backtest", "🚀 Optimizer (Parameter Sweep)"])

    if st.session_state['stock_data'] is not None:
        df = st.session_state['stock_data']
        
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
            'trend_sma': trend_sma,
            'confirmation_days': confirmation_days,
            'bb_window': bb_window,
            'bb_std': bb_std,
            'interest_rate_pct': interest_rate_pct,
            'enable_dividends': enable_dividends,
            'restrict_ex_date': restrict_ex_date,
            'enable_taxes': enable_taxes,
            'st_tax_rate': st_tax_rate,
            'lt_tax_rate': lt_tax_rate,
            'initial_investment': initial_investment,
            'monthly_investment': monthly_investment,
            'trade_size_type': trade_size_type,
            'shares_per_trade': shares_per_trade,
            'min_trade_amt': min_trade_amt,
            'max_trade_amt': max_trade_amt
        }

        with tab1:
            if st.button("Run Single Backtest"):
                res = run_simulation(df, current_params)
                st.session_state['sim_results'] = res
                
                baseline_params = current_params.copy()
                baseline_params['strategy_mode'] = "Dip Accumulation"
                baseline_params['buy_drop_pct'] = baseline_buy_drop_pct
                res_base = run_simulation(df, baseline_params)
                st.session_state['baseline_results'] = res_base
            
            if st.session_state['sim_results'] is not None and st.session_state['baseline_results'] is not None:
                res = st.session_state['sim_results']
                res_base = st.session_state['baseline_results']
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Final Value", f"{currency_symbol}{res['final_value']:,.2f}", delta=f"Inv: {currency_symbol}{res['invested_capital']:,.0f}")
                c2.metric("Selected Strategy XIRR", f"{res['strategy_xirr']:.2%}")
                c3.metric("Baseline Dip XIRR", f"{res_base['strategy_xirr']:.2%}")
                c4.metric("Buy & Hold XIRR", f"{res['bh_xirr']:.2%}")
                c5.metric("Trade Efficiency", f"{res['trade_xirr']:.2%}")
                
                c6, c7, c8, c9 = st.columns(4)
                c6.metric("Trades", f"{res['executed_trades']} / {res['executed_trades'] + res['missed_trades']}")
                c7.metric("Passive Income", f"{currency_symbol}{res['passive_income']:,.2f}")
                c8.metric("Cash Balance", f"{currency_symbol}{res['wallet_cash']:,.2f}")
                c9.metric("Taxes Paid", f"{currency_symbol}{res['total_taxes_paid']:,.2f}")

                if enable_taxes and res['tax_events']:
                    with st.expander(f"🏛️ Tax Payment Schedule ({len(res['tax_events'])})"):
                        st.dataframe(pd.DataFrame(res['tax_events']))

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
                        s_price = t['sell_price'] if t.get('status') in ["Closed", "VA Sell", "Tax Liquidation"] else last_close
                        p_share = s_price - t['buy_price']
                    
                    logs.append({
                        "Date": t['buy_date'].strftime('%Y-%m-%d') if pd.notnull(t['buy_date']) else "-",
                        "Buy": f"{currency_symbol}{t['buy_price']:.2f}",
                        "Sell": f"{currency_symbol}{s_price:.2f}" if decision == "Executed" else "-",
                        "Qty": f"{qty:.4f}" if decision == "Executed" else "-",
                        "Profit": f"{currency_symbol}{p_share * qty:.2f}" if decision == "Executed" else "-",
                        "Status": t.get('status', 'Unknown'),
                        "Drop/Signal": t.get('drop_pct', '')
                    })
                st.dataframe(pd.DataFrame(logs), use_container_width=True)
                
                st.subheader("📈 Portfolio Growth Over Time")
                
                hist_df = pd.DataFrame(res['daily_history'])
                hist_df.rename(columns={"Total Value": "Selected Strategy"}, inplace=True)
                
                base_df = pd.DataFrame(res_base['daily_history'])[['Date', 'Total Value']].copy()
                base_df.rename(columns={"Total Value": "Baseline (Dip Accum.)"}, inplace=True)
                
                chart_df = pd.merge(hist_df, base_df, on="Date")
                chart_df['Date'] = pd.to_datetime(chart_df['Date'])
                
                all_metrics = ['Selected Strategy', 'Baseline (Dip Accum.)', 'Buy & Hold', 'Cash', 'Open Positions']
                chart_data_melted = chart_df.melt(id_vars='Date', value_vars=all_metrics, var_name='Metric', value_name='Value')
                
                selected_metrics = st.multiselect("Select Metrics:", options=all_metrics, default=all_metrics[:3])
                
                if selected_metrics:
                    filtered_chart_df = chart_data_melted[chart_data_melted['Metric'].isin(selected_metrics)]
                    
                    chart = alt.Chart(filtered_chart_df).mark_line().encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Value:Q', title=f'Value ({currency_symbol})'),
                        color=alt.Color('Metric:N', legend=alt.Legend(title="Metrics")),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Metric:N', alt.Tooltip('Value:Q', format=',.2f')]
                    )
                    
                    marker_data = []
                    val_lookup = hist_df.set_index('Date')['Selected Strategy'].to_dict()
                    
                    for t in res['trades']:
                        if res['decisions'].get(t['trade_id']) == "Executed":
                            buy_d = pd.to_datetime(t['buy_date'])
                            if buy_d in val_lookup:
                                marker_data.append({"Date": buy_d, "Action": "Buy", "Value": val_lookup[buy_d], "Stock Price": t['buy_price']})
                            
                            if t.get('status') in ["Closed", "VA Sell", "Tax Liquidation"]:
                                sell_d = pd.to_datetime(t['sell_date'])
                                if pd.notnull(sell_d) and sell_d in val_lookup:
                                    marker_data.append({"Date": sell_d, "Action": t.get('status'), "Value": val_lookup[sell_d], "Stock Price": t['sell_price']})
                    
                    if marker_data and 'Selected Strategy' in selected_metrics:
                        markers_df = pd.DataFrame(marker_data)
                        markers = alt.Chart(markers_df).mark_circle(size=80, opacity=1).encode(
                            x=alt.X('Date:T'),
                            y=alt.Y('Value:Q'),
                            color=alt.Color('Action:N', scale=alt.Scale(domain=['Buy', 'Closed', 'VA Sell', 'Tax Liquidation'], range=['#00b050', '#ff0000', '#FFA500', '#800080']), legend=None),
                            tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Action:N', alt.Tooltip('Stock Price:Q', format=',.2f'), alt.Tooltip('Value:Q', format=',.2f')]
                        )
                        final_chart = alt.layer(chart, markers).resolve_scale(color='independent').properties(height=400).interactive()
                    else:
                        final_chart = chart.properties(height=400).interactive()
                    
                    st.altair_chart(final_chart, use_container_width=True)
                else:
                    st.warning("Please select at least one metric to display the chart.")

        with tab2:
            st.write("Automatically test different parameters to find the 'Sweet Spot'.")
            
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                optimize_target = st.selectbox("Parameter to Optimize", 
                                             ["Trailing Stop %", "Activation Target %", "Buy Drop %", "SMA Short", "RSI Buy", "Trend SMA", "Confirmation Days", "BB Window", "BB Std Dev"])
            with opt_col2:
                st.write("Range Settings")
                r_start = st.number_input("Start", value=1.0, step=0.5)
                r_end = st.number_input("End", value=10.0, step=0.5)
                r_step = st.number_input("Step", value=0.5, step=0.1)

            if st.button("Run Optimization Sweep"):
                if r_step <= 0:
                    st.error("Step size must be greater than 0.")
                else:
                    results_sweep = []
                    test_values = np.arange(r_start, r_end + 0.001, r_step)
                    bar = st.progress(0)
                    best_xirr = -999.0
                    best_val = 0.0
                    
                    for idx, val in enumerate(test_values):
                        temp_params = current_params.copy()
                        if optimize_target == "Trailing Stop %": temp_params['use_trailing_stop'] = True; temp_params['trailing_stop_pct'] = val
                        elif optimize_target == "Activation Target %": temp_params['sell_profit_pct'] = val
                        elif optimize_target == "Buy Drop %": temp_params['buy_drop_pct'] = val
                        elif optimize_target == "SMA Short": temp_params['sma_short'] = int(val)
                        elif optimize_target == "RSI Buy": temp_params['rsi_buy'] = int(val)
                        elif optimize_target == "Trend SMA": temp_params['trend_sma'] = int(val)
                        elif optimize_target == "Confirmation Days": temp_params['confirmation_days'] = int(val)
                        elif optimize_target == "BB Window": temp_params['bb_window'] = int(val)
                        elif optimize_target == "BB Std Dev": temp_params['bb_std'] = val
                        
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

except Exception as e:
    st.error("🚨 **A critical error occurred while building the app.**")
    st.error(f"Error Details: `{e}`")
    st.code(traceback.format_exc())
