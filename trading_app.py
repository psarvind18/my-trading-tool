import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import math

# --- Configuration ---
st.set_page_config(page_title="Advanced Trading Backtest", layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def xirr(transactions):
    """
    Calculates XIRR (Extended Internal Rate of Return) using Newton-Raphson method.
    transactions: list of tuples [(date, amount), ...]
    amount should be negative for outflows (investments) and positive for inflows (returns).
    """
    if not transactions:
        return 0.0
    
    # Sort by date
    transactions.sort(key=lambda x: x[0])
    
    start_date = transactions[0][0]
    
    # Check if we have both positive and negative cash flows
    amounts = [t[1] for t in transactions]
    if all(a >= 0 for a in amounts) or all(a <= 0 for a in amounts):
        return 0.0
        
    def npv(rate):
        total_npv = 0.0
        for dt, amt in transactions:
            days = (dt - start_date).days
            total_npv += amt / ((1 + rate) ** (days / 365.0))
        return total_npv

    def npv_derivative(rate):
        d_npv = 0.0
        for dt, amt in transactions:
            days = (dt - start_date).days
            d_npv -= (days / 365.0) * amt / ((1 + rate) ** ((days / 365.0) + 1))
        return d_npv

    # Newton-Raphson
    rate = 0.1 # Initial guess 10%
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
st.title("🌏 Professional Trading Backtester")
st.markdown("""
**Metrics Explained:**
* **CAGR:** Compound Annual Growth Rate of your *entire* initial investment.
* **Portfolio IRR:** Internal Rate of Return of the wallet (Matches CAGR if no deposits/withdrawals occur).
* **Trade XIRR:** The efficiency of the capital *actually used*. A high Trade XIRR means your money works hard when deployed, even if the Portfolio CAGR is low due to cash drag.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Market Data")
    ticker_symbol = st.text_input("Ticker Symbol", value="NIFTYBEES.NS").upper()
    currency_symbol = st.text_input("Currency Symbol", value="₹")
    
    today = date.today()
    start_input = st.date_input("Start Date", value=date(today.year - 1, 1, 1))
    end_input = st.date_input("End Date", value=today)
    
    st.divider()
    
    st.header("2. Strategy Settings")
    initial_investment = st.number_input(f"Initial Investment ({currency_symbol})", value=100000.0, step=500.0)
    shares_per_trade = st.number_input("Shares per Trade", min_value=1, value=10, step=1)
    
    run_btn = st.button("Run Backtest")

# --- Execution ---
if run_btn or 'data_loaded' in st.session_state:
    st.session_state['data_loaded'] = True
    
    with st.spinner(f'Analyzing {ticker_symbol}...'):
        df = load_stock_data(ticker_symbol, start_input, end_input)

    if df.empty or len(df) < 2:
        st.error("No sufficient data found.")
    else:
        # --- 1. Signal Generation (Tiered) ---
        potential_trades = []
        for i in range(1, len(df)):
            prev_close = df.loc[i-1, 'Close']
            daily_low = df.loc[i, 'Low']
            daily_date = df.loc[i, 'Date']
            
            drop_level = 1
            while True:
                target_buy_price = prev_close * (1 - (0.01 * drop_level))
                if daily_low <= target_buy_price:
                    # Trade Parameters
                    buy_price = target_buy_price
                    target_sell_price = buy_price * 1.04
                    
                    sell_date = pd.NaT
                    sell_price = 0.0
                    status = "Open"
                    
                    for j in range(i + 1, len(df)):
                        if df.loc[j, 'High'] >= target_sell_price:
                            sell_date = df.loc[j, 'Date']
                            sell_price = target_sell_price
                            status = "Closed"
                            break
                    
                    potential_trades.append({
                        "trade_id": len(potential_trades),
                        "buy_date": daily_date,
                        "buy_price": buy_price,
                        "drop_pct": f"{drop_level}%",
                        "sell_date": sell_date,
                        "sell_price": sell_price,
                        "status": status
                    })
                    drop_level += 1
                else:
                    break
        
        # --- 2. Wallet Simulation ---
        events = []
        for t in potential_trades:
            cost = t['buy_price'] * shares_per_trade
            revenue = t['sell_price'] * shares_per_trade
            
            # Buy Event
            events.append({"date": t['buy_date'], "type": "buy", "trade_id": t['trade_id'], "amount": cost})
            # Sell Event
            if t['status'] == "Closed":
                events.append({"date": t['sell_date'], "type": "sell", "trade_id": t['trade_id'], "amount": revenue})
        
        # Sort (Sell before Buy on same day to recycle cash)
        type_priority = {'sell': 0, 'buy': 1}
        events.sort(key=lambda x: (x['date'], type_priority[x['type']]))
        
        wallet = initial_investment
        active_holdings = set() 
        trade_decisions = {}
        
        # For Trade XIRR Calculation: Track flows of executed trades ONLY
        trade_cash_flows = []
        
        executed_count = 0
        missed_count = 0
        
        for event in events:
            tid = event['trade_id']
            if event['type'] == 'buy':
                if wallet >= event['amount']:
                    wallet -= event['amount']
                    active_holdings.add(tid)
                    trade_decisions[tid] = "Executed"
                    executed_count += 1
                    
                    # Record flow for Trade XIRR (Investment is negative)
                    trade_cash_flows.append((event['date'], -event['amount']))
                else:
                    trade_decisions[tid] = "Missed"
                    missed_count += 1
            elif event['type'] == 'sell':
                if tid in active_holdings:
                    wallet += event['amount']
                    active_holdings.remove(tid)
                    
                    # Record flow for Trade XIRR (Return is positive)
                    trade_cash_flows.append((event['date'], event['amount']))

        # --- 3. Final Valuation ---
        last_close_price = df.iloc[-1]['Close']
        open_position_value = 0
        
        # Calculate Open Positions Value
        # Also finish Trade XIRR flows for open positions
        final_date = df.iloc[-1]['Date']
        
        for t in potential_trades:
            if trade_decisions.get(t['trade_id']) == "Executed" and t['status'] == "Open":
                current_val = last_close_price * shares_per_trade
                open_position_value += current_val
                # Mark to market for XIRR
                trade_cash_flows.append((final_date, current_val))

        final_portfolio_value = wallet + open_position_value
        
        # --- 4. Metric Calculations ---
        
        # A. CAGR
        total_days = (end_input - start_input).days
        years = total_days / 365.25
        cagr = 0.0
        if years > 0 and initial_investment > 0:
            cagr = (final_portfolio_value / initial_investment) ** (1 / years) - 1
            
        # B. Portfolio IRR (Cash Flow of the Wallet)
        # Flows: Initial Investment (Neg), Final Value (Pos)
        portfolio_flows = [
            (pd.Timestamp(start_input), -initial_investment),
            (pd.Timestamp(end_input), final_portfolio_value)
        ]
        portfolio_irr = xirr(portfolio_flows)
        
        # C. Trade IRR (Capital Efficiency)
        trade_irr = xirr(trade_cash_flows)

        # --- Dashboard ---
        st.markdown("### 📊 Performance Metrics")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Value", f"{currency_symbol}{final_portfolio_value:,.2f}", 
                  delta=f"{currency_symbol}{final_portfolio_value - initial_investment:,.2f}")
        c2.metric("CAGR", f"{cagr:.2%}")
        c3.metric("Portfolio IRR", f"{portfolio_irr:.2%}", help="Return on the total wallet.")
        c4.metric("Trade XIRR", f"{trade_irr:.2%}", help="Return on capital actually employed in trades.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Executed Trades", executed_count)
        c6.metric("Missed Trades", missed_count, delta_color="inverse")
        c7.metric("Cash Balance", f"{currency_symbol}{wallet:,.2f}")
        c8.metric("Open Pos Value", f"{currency_symbol}{open_position_value:,.2f}")
        
        # --- Table ---
        st.markdown("### 📝 Detailed Trade Log")
        
        final_log = []
        for t in potential_trades:
            decision = trade_decisions.get(t['trade_id'], "Missed")
            profit = 0.0
            sell_price_display = 0.0
            
            if decision == "Executed":
                if t['status'] == "Closed":
                    profit = (t['sell_price'] - t['buy_price']) * shares_per_trade
                    sell_price_display = t['sell_price']
                else:
                    profit = (last_close_price - t['buy_price']) * shares_per_trade
                    sell_price_display = last_close_price
            
            final_log.append({
                "Buy Date": t['buy_date'].strftime('%Y-%m-%d'),
                "Tier": t['drop_pct'],
                "Buy Price": t['buy_price'],
                "Sell Date": t['sell_date'].strftime('%Y-%m-%d') if pd.notnull(t['sell_date']) else "Open",
                "Sell Price": sell_price_display if decision == "Executed" else 0,
                "Status": t['status'],
                "Profit": profit if decision == "Executed" else 0.0,
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
        fmt_df['Profit'] = fmt_df['Profit'].map(lambda x: f"{currency_symbol}{x:,.2f}")
        
        st.dataframe(fmt_df.style.apply(style_rows, axis=1), use_container_width=True)
else:
    st.info("👈 Enter settings and click **Run Backtest**.")
