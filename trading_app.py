import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(page_title="SIP Trading Backtester", layout="wide")

# --- Helper Functions ---
def fetch_data_from_yahoo(ticker_symbol, start_date, end_date):
    """
    Fetches data only when explicitly called.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        df.index = df.index.tz_localize(None)
        df.reset_index(inplace=True)
        
        if 'Dividends' not in df.columns:
            df['Dividends'] = 0.0
            
        return df, ticker.info
    except Exception as e:
        return None, str(e)

def xirr(transactions):
    if not transactions:
        return 0.0
    transactions.sort(key=lambda x: x[0])
    amounts = [t[1] for t in transactions]
    if all(a >= 0 for a in amounts) or all(a <= 0 for a in amounts):
        return 0.0
    
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
            if abs(new_rate - rate) < 1e-6:
                return new_rate
            rate = new_rate
        except:
            return 0.0
    return rate

# --- Main App ---
st.title("🛡️ SIP & Trading Strategy Backtester")

# --- Initialize Session State for Data ---
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None
if 'stock_info' not in st.session_state:
    st.session_state['stock_info'] = {}

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Loading")
    ticker_symbol = st.text_input("Ticker Symbol", value="VOO").upper()
    
    today = date.today()
    start_input = st.date_input("Start Date", value=date(today.year - 1, 1, 1))
    end_input = st.date_input("End Date", value=today)
    
    # SEPARATE BUTTON TO LOAD DATA
    if st.button("Step 1: Fetch Data"):
        with st.spinner(f"Downloading {ticker_symbol}..."):
            df, info = fetch_data_from_yahoo(ticker_symbol, start_input, end_input)
            if df is not None and not df.empty:
                st.session_state['stock_data'] = df
                st.session_state['stock_info'] = info
                st.session_state['data_ticker'] = ticker_symbol # Remember what we downloaded
                st.success(f"Loaded {len(df)} days for {ticker_symbol}!")
            else:
                st.error(f"Failed to load data. Error: {info}")

    st.divider()
    
    st.header("2. Strategy Settings")
    # Strategy Inputs
    strategy_mode = st.radio("Strategy Type:", ("Swing Trading", "Dip Accumulation"))
    buy_drop_pct = st.number_input("Buy Drop Step (%)", min_value=0.1, value=1.0, step=0.1)
    
    if strategy_mode == "Swing Trading":
        sell_profit_pct = st.number_input("Sell Profit Target (%)", min_value=0.1, value=4.0, step=0.1)
    else:
        sell_profit_pct = 0.0
    
    st.divider()
    
    st.header("3. Cash & Yield")
    interest_rate_pct = st.number_input("Cash Interest Rate (%)", min_value=0.0, value=4.5, step=0.1)
    enable_dividends = st.checkbox("Include Dividends", value=True)
    restrict_ex_date = st.checkbox("Restrict Trade around Ex-Date", value=True) if enable_dividends else False
    
    st.divider()
    
    st.header("4. Investment Plan")
    currency_symbol = st.text_input("Currency Symbol", value="$")
    initial_investment = st.number_input(f"Initial Investment", value=10000.0, step=500.0)
    monthly_investment = st.number_input(f"Monthly Contribution", value=0.0, step=100.0)
    shares_per_trade = st.number_input("Shares per Trade", min_value=1, value=1, step=1)
    
    calc_btn = st.button("Step 2: Run Backtest")

# --- Execution ---
if st.session_state['stock_data'] is not None:
    # Use data from memory
    df = st.session_state['stock_data']
    info = st.session_state['stock_info']
    
    # Display what data is currently loaded
    st.info(f"Using loaded data for **{st.session_state.get('data_ticker', 'Unknown')}** ({len(df)} records). If you change Ticker/Dates, click 'Fetch Data' again.")

    if calc_btn:
        # --- Pre-Calculation: Restricted Days ---
        restricted_indices = set()
        dividend_events = []
        
        if enable_dividends:
            div_indices = df.index[df['Dividends'] > 0].tolist()
            for idx in div_indices:
                dividend_events.append({
                    "Date": df.loc[idx, 'Date'],
                    "Amount": df.loc[idx, 'Dividends']
                })
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
            
            if i in restricted_indices:
                continue 
            
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
                        target_sell_price = buy_price * (1 + (sell_profit_pct / 100.0))
                        for j in range(i + 1, len(df)):
                            if j in restricted_indices: continue
                            if df.loc[j, 'High'] >= target_sell_price:
                                sell_date = df.loc[j, 'Date']
                                sell_price = target_sell_price
                                status = "Closed"
                                break
                    
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
                        if sell_date not in trades_by_date:
                            trades_by_date[sell_date] = {'buys': [], 'sells': []}
                        trades_by_date[sell_date]['sells'].append(trade_obj)
                    
                    drop_level += 1
                else:
                    break
        
        # --- 2. Simulation ---
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
        
        # B&H Benchmarking
        bh_shares = 0
        bh_wallet = initial_investment
        bh_cash_flows = [ (df.iloc[0]['Date'], -initial_investment) ]
        
        first_open_price = df.iloc[0]['Open'] 
        start_shares = int(bh_wallet // first_open_price)
        bh_shares += start_shares
        bh_wallet -= start_shares * first_open_price

        prev_sim_date = df.iloc[0]['Date']
        last_month_processed = -1
        
        for i in range(len(df)):
            curr_date = df.iloc[i]['Date']
            curr_close = df.iloc[i]['Close']
            
            # Monthly Contribution
            if curr_date.month != last_month_processed:
                if monthly_investment > 0 and i > 0:
                    # Strategy Wallet
                    wallet += monthly_investment
                    portfolio_cash_flows.append((curr_date, -monthly_investment))
                    total_invested_capital += monthly_investment
                    
                    # B&H Wallet
                    bh_wallet += monthly_investment
                    bh_cash_flows.append((curr_date, -monthly_investment))
                    new_shares = int(bh_wallet // curr_close)
                    if new_shares > 0:
                        bh_shares += new_shares
                        bh_wallet -= new_shares * curr_close
                
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
                    if len(active_holdings) > 0:
                        total_held = len(active_holdings) * shares_per_trade
                        payout = total_held * today_div_amount
                        wallet += payout
                        total_dividends_earned += payout
                    if bh_shares > 0:
                        bh_payout = bh_shares * today_div_amount
                        bh_wallet += bh_payout

            # Trading
            if curr_date in trades_by_date:
                day_activity = trades_by_date[curr_date]
                for t in day_activity['sells']:
                    if t['trade_id'] in active_holdings:
                        revenue = t['sell_price'] * shares_per_trade
                        wallet += revenue
                        active_holdings.remove(t['trade_id'])
                        trade_cash_flows.append((curr_date, revenue))
                
                for t in day_activity['buys']:
                    cost = t['buy_price'] * shares_per_trade
                    if wallet >= cost:
                        wallet -= cost
                        active_holdings.add(t['trade_id'])
                        trade_decisions[t['trade_id']] = "Executed"
                        executed_count += 1
                        trade_cash_flows.append((curr_date, -cost))
                    else:
                        trade_decisions[t['trade_id']] = "Missed"
                        missed_count += 1
            
            prev_sim_date = curr_date

        # --- 4. Final Valuation ---
        last_close_price = df.iloc[-1]['Close']
        final_date = df.iloc[-1]['Date']
        
        open_position_value = 0
        for t in potential_trades:
            if trade_decisions.get(t['trade_id']) == "Executed" and t['status'] == "Open":
                val = last_close_price * shares_per_trade
                open_position_value += val
                trade_cash_flows.append((final_date, val))

        final_strategy_value = wallet + open_position_value
        portfolio_cash_flows.append((final_date, final_strategy_value))
        
        final_bh_value = (bh_shares * last_close_price) + bh_wallet
        bh_cash_flows.append((final_date, final_bh_value))

        # --- 5. Metrics ---
        strategy_xirr = xirr(portfolio_cash_flows)
        bh_xirr = xirr(bh_cash_flows)
        trade_efficiency_xirr = xirr(trade_cash_flows)
        
        curr_yield_display = "N/A"
        if 'dividendYield' in info and info['dividendYield'] is not None:
            curr_yield_display = f"{info['dividendYield']*100:.2f}%"

        # --- Dashboard ---
        st.markdown(f"### 📊 Performance: {strategy_mode} (Yield: {curr_yield_display})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Wallet Value", f"{currency_symbol}{final_strategy_value:,.2f}", 
                  delta=f"Invested: {currency_symbol}{total_invested_capital:,.0f}")
        c2.metric("Strategy Return (XIRR)", f"{strategy_xirr:.2%}")
        c3.metric("Buy & Hold Return (XIRR)", f"{bh_xirr:.2%}")
        c4.metric("Capital Efficiency (Trade XIRR)", f"{trade_efficiency_xirr:.2%}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Executed Trades", executed_count)
        c6.metric("Total Passive Income", f"{currency_symbol}{total_interest_earned + total_dividends_earned:,.2f}")
        c7.metric("Cash Balance", f"{currency_symbol}{wallet:,.2f}")
        c8.metric("Open Pos Value", f"{currency_symbol}{open_position_value:,.2f}")
        
        if enable_dividends and dividend_events:
             with st.expander(f"📅 Dividend Schedule ({len(dividend_events)} events)"):
                st.dataframe(pd.DataFrame(dividend_events), use_container_width=True)

        st.markdown("### 📝 Detailed Trade Log")
        
        final_log = []
        for t in potential_trades:
            decision = trade_decisions.get(t['trade_id'], "Missed")
            profit_total = 0.0
            profit_per_share = 0.0
            sell_price_disp = 0.0
            
            if decision == "Executed":
                if t['status'] == "Closed":
                    profit_per_share = t['sell_price'] - t['buy_price']
                    sell_price_disp = t['sell_price']
                else:
                    profit_per_share = last_close_price - t['buy_price']
                    sell_price_disp = last_close_price
                profit_total = profit_per_share * shares_per_trade
            
            final_log.append({
                "Buy Date": t['buy_date'].strftime('%Y-%m-%d'),
                "Drop %": t['drop_pct'],
                "Buy Price": t['buy_price'],
                "Sell Date": t['sell_date'].strftime('%Y-%m-%d') if pd.notnull(t['sell_date']) else "Open",
                "Sell Price": sell_price_disp if decision == "Executed" else 0,
                "Profit/Share": profit_per_share if decision == "Executed" else 0.0,
                "Total P&L": profit_total if decision == "Executed" else 0.0,
                "Status": t['status'],
                "Execution": decision
            })
            
        results_df = pd.DataFrame(final_log)
        
        def style_rows(row):
            if row['Execution'] == 'Missed':
                return ['background-color: #ffebee; color: #c62828'] * len(row)
            elif row['Status'] == 'Open':
                return ['background-color: #e3f2fd; color: #0d47a1'] * len(row)
            else:
                return ['background-color: #e8f5e9; color: #1b5e20'] * len(row)
        
        fmt_df = results_df.copy()
        fmt_df['Buy Price'] = fmt_df['Buy Price'].map(lambda x: f"{currency_symbol}{x:,.2f}")
        fmt_df['Sell Price'] = fmt_df['Sell Price'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if x > 0 else "-")
        fmt_df['Profit/Share'] = fmt_df['Profit/Share'].map(lambda x: f"{currency_symbol}{x:,.2f}")
        fmt_df['Total P&L'] = fmt_df['Total P&L'].map(lambda x: f"{currency_symbol}{x:,.2f}")
        
        st.dataframe(fmt_df.style.apply(style_rows, axis=1), use_container_width=True)

else:
    st.info("👈 Please click **'Step 1: Fetch Data'** in the sidebar to begin.")
