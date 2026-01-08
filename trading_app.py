import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(page_title="SIP Trading Backtester", layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_stock_data(ticker_symbol, start_date, end_date):
    """
    Fetches OHLC data AND Dividend/Split events.
    Uses auto_adjust=False to get actual market prices.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        # Ensure timezone naive
        df.index = df.index.tz_localize(None)
        df.reset_index(inplace=True)
        
        if 'Dividends' not in df.columns:
            df['Dividends'] = 0.0
            
        return df, ticker.info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), {}

def xirr(transactions):
    """
    Calculates XIRR (Extended Internal Rate of Return).
    transactions: list of tuples [(date, amount)]
    - Investments are NEGATIVE
    - Final Value / Returns are POSITIVE
    """
    if not transactions:
        return 0.0
    transactions.sort(key=lambda x: x[0])
    
    # Validation: Must have at least one negative and one positive
    amounts = [t[1] for t in transactions]
    if all(a >= 0 for a in amounts) or all(a <= 0 for a in amounts):
        return 0.0
        
    start_date = transactions[0][0]
        
    def npv(rate):
        total_npv = 0.0
        for dt, amt in transactions:
            days = (dt - start_date).days
            # Protect against division by zero or extreme rates
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

# --- Sidebar ---
with st.sidebar:
    st.header("1. Strategy Selection")
    strategy_mode = st.radio(
        "Choose Strategy Type:",
        ("Swing Trading", "Dip Accumulation"),
        help="Swing: Buy Dip & Sell High. \nAccumulation: Buy Dip & Hold Forever."
    )
    
    st.divider()
    
    st.header("2. Market Data")
    ticker_symbol = st.text_input("Ticker Symbol", value="VOO").upper()
    currency_symbol = st.text_input("Currency Symbol", value="$")
    
    today = date.today()
    start_input = st.date_input("Start Date", value=date(today.year - 1, 1, 1))
    end_input = st.date_input("End Date", value=today)
    
    st.divider()
    
    st.header("3. Trade Settings")
    buy_drop_pct = st.number_input("Buy Drop Step (%)", min_value=0.1, value=1.0, step=0.1)
    
    if strategy_mode == "Swing Trading":
        sell_profit_pct = st.number_input("Sell Profit Target (%)", min_value=0.1, value=4.0, step=0.1)
    else:
        sell_profit_pct = 0.0
        st.info("Selling disabled in Accumulation mode.")
    
    st.divider()
    
    st.header("4. Cash & Dividends")
    interest_rate_pct = st.number_input("Cash Interest Rate (%)", min_value=0.0, value=4.5, step=0.1)
    
    enable_dividends = st.checkbox("Include Dividends", value=True)
    if enable_dividends:
        restrict_ex_date = st.checkbox("Restrict Trade around Ex-Date", value=True)
    else:
        restrict_ex_date = False
    
    st.divider()
    
    st.header("5. Investment Plan")
    initial_investment = st.number_input(f"Initial Investment ({currency_symbol})", value=10000.0, step=500.0)
    monthly_investment = st.number_input(f"Monthly Contribution ({currency_symbol})", value=0.0, step=100.0, help="Added on the first trading day of each month.")
    shares_per_trade = st.number_input("Shares per Trade", min_value=1, value=1, step=1)
    
    run_btn = st.button("Run Backtest")

# --- Execution ---
if run_btn or 'data_loaded' in st.session_state:
    st.session_state['data_loaded'] = True
    
    with st.spinner(f'Processing {ticker_symbol}...'):
        df, info = load_stock_data(ticker_symbol, start_input, end_input)

    if df.empty or len(df) < 5:
        st.error("No sufficient data found.")
    else:
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
        
        # --- 2. Simulation Setup ---
        
        # A. Strategy Portfolio
        wallet = initial_investment
        active_holdings = set()
        trade_decisions = {}
        trade_cash_flows = [] # For Trade XIRR (Efficiency)
        portfolio_cash_flows = [ (df.iloc[0]['Date'], -initial_investment) ] # For Portfolio XIRR
        
        total_interest_earned = 0.0
        total_dividends_earned = 0.0
        total_invested_capital = initial_investment
        
        executed_count = 0
        missed_count = 0
        
        # B. Buy & Hold (SIP) Benchmark
        bh_shares = 0
        bh_wallet = initial_investment
        bh_cash_flows = [ (df.iloc[0]['Date'], -initial_investment) ]
        
        # Initial Buy for B&H
        first_open_price = df.iloc[0]['Open'] # Or Close? Usually Close is easier or Open of first day
        # Let's use Open of first day to simulate entering immediately
        start_shares = int(bh_wallet // first_open_price)
        bh_shares += start_shares
        bh_wallet -= start_shares * first_open_price

        # Simulation Loop
        prev_sim_date = df.iloc[0]['Date']
        last_month_processed = -1
        
        for i in range(len(df)):
            curr_date = df.iloc[i]['Date']
            curr_close = df.iloc[i]['Close'] # For B&H buying
            
            # --- Monthly Contribution Logic ---
            # Check if we moved to a new month (simple logic: current month != last processed month)
            # This ensures it happens on the FIRST trading day of the month found in data
            if curr_date.month != last_month_processed:
                if i > 0: # Don't double add on Day 1 if it's start of dataset (already handled by initial)
                     # Wait, initial is initial. Monthly is Monthly.
                     # Usually Monthly starts month 2 or month 1? 
                     # Let's assume Month 1 contribution is the Initial Investment.
                     # So we start adding from the NEXT month change.
                     # OR: If user wants monthly add, they usually mean NEXT month or immediate?
                     # Standard backtest: Initial is Day 0. Monthly starts next month or same month?
                     # Let's simply add IF it's not the very first day of data to avoid double counting "Initial" as "Monthly"
                     if monthly_investment > 0:
                        # 1. Strategy Wallet
                        wallet += monthly_investment
                        portfolio_cash_flows.append((curr_date, -monthly_investment))
                        total_invested_capital += monthly_investment
                        
                        # 2. Buy & Hold Wallet (SIP)
                        bh_wallet += monthly_investment
                        bh_cash_flows.append((curr_date, -monthly_investment))
                        # Immediate Buy for B&H
                        new_shares = int(bh_wallet // curr_close)
                        if new_shares > 0:
                            bh_shares += new_shares
                            bh_wallet -= new_shares * curr_close
                
                last_month_processed = curr_date.month

            # --- 1. Daily Interest ---
            days_delta = (curr_date - prev_sim_date).days
            if days_delta > 0:
                if wallet > 0:
                    interest = wallet * (interest_rate_pct / 100.0 / 365.0) * days_delta
                    wallet += interest
                    total_interest_earned += interest
                # Interest for B&H cash?
                if bh_wallet > 0:
                    bh_interest = bh_wallet * (interest_rate_pct / 100.0 / 365.0) * days_delta
                    bh_wallet += bh_interest

            # --- 2. Dividends ---
            if enable_dividends:
                today_div_amount = df.loc[i, 'Dividends']
                if today_div_amount > 0:
                    # Strategy
                    if len(active_holdings) > 0:
                        total_held = len(active_holdings) * shares_per_trade
                        payout = total_held * today_div_amount
                        wallet += payout
                        total_dividends_earned += payout
                    
                    # Buy & Hold
                    if bh_shares > 0:
                        bh_payout = bh_shares * today_div_amount
                        bh_wallet += bh_payout

            # --- 3. Trading Activity (Strategy Only) ---
            if curr_date in trades_by_date:
                day_activity = trades_by_date[curr_date]
                
                # Sells
                for t in day_activity['sells']:
                    if t['trade_id'] in active_holdings:
                        revenue = t['sell_price'] * shares_per_trade
                        wallet += revenue
                        active_holdings.remove(t['trade_id'])
                        trade_cash_flows.append((curr_date, revenue))
                
                # Buys
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
        
        # A. Strategy Value
        open_position_value = 0
        for t in potential_trades:
            if trade_decisions.get(t['trade_id']) == "Executed" and t['status'] == "Open":
                val = last_close_price * shares_per_trade
                open_position_value += val
                trade_cash_flows.append((final_date, val)) # Mark-to-market for Trade XIRR

        final_strategy_value = wallet + open_position_value
        # Add final value to portfolio flows for XIRR
        portfolio_cash_flows.append((final_date, final_strategy_value))
        
        # B. Buy & Hold Value
        final_bh_value = (bh_shares * last_close_price) + bh_wallet
        bh_cash_flows.append((final_date, final_bh_value))

        # --- 5. Metrics Calculation ---
        
        # Portfolio XIRR (Strategy)
        strategy_xirr = xirr(portfolio_cash_flows)
        
        # Benchmark XIRR (Buy & Hold)
        bh_xirr = xirr(bh_cash_flows)
        
        # Trade XIRR (Efficiency)
        trade_efficiency_xirr = xirr(trade_cash_flows)
        
        # Yield Display
        curr_yield_display = "N/A"
        if 'dividendYield' in info and info['dividendYield'] is not None:
            curr_yield_display = f"{info['dividendYield']*100:.2f}%"

        # --- Dashboard ---
        st.markdown(f"### 📊 Performance Summary (Yield: {curr_yield_display})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Wallet Value", f"{currency_symbol}{final_strategy_value:,.2f}", 
                  delta=f"Invested: {currency_symbol}{total_invested_capital:,.0f}")
        
        c2.metric("Strategy Return (XIRR)", f"{strategy_xirr:.2%}", 
                  help="Annualized return including all monthly contributions.")
        
        c3.metric("Buy & Hold Return (XIRR)", f"{bh_xirr:.2%}", 
                  help="Benchmark: Investing Initial + Monthly amounts into the stock immediately.")
        
        c4.metric("Capital Efficiency (Trade XIRR)", f"{trade_efficiency_xirr:.2%}",
                  help="Return on the specific capital used in trades (excluding idle cash drag).")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Executed Trades", executed_count)
        c6.metric("Total Passive Income", f"{currency_symbol}{total_interest_earned + total_dividends_earned:,.2f}")
        c7.metric("Cash Balance", f"{currency_symbol}{wallet:,.2f}")
        c8.metric("Open Pos Value", f"{currency_symbol}{open_position_value:,.2f}")
        
        # --- Log ---
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
    st.info("👈 Enter settings and click **Run Backtest**.")
