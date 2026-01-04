import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

# --- Configuration ---
st.set_page_config(page_title="Customizable Trading Backtest", layout="wide")

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
    Calculates XIRR (Extended Internal Rate of Return).
    transactions: list of tuples [(date, amount), ...]
    """
    if not transactions:
        return 0.0
    
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
            # Avoid division by zero or complex numbers with negative rates
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
st.title("🛠️ Fully Customizable Trading Backtester")

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
    
    # New Inputs for Buy/Sell Logic
    buy_drop_pct = st.number_input(
        "Buy Drop Step (%)", 
        min_value=0.1, 
        value=1.0, 
        step=0.1,
        help="Buy at every X% drop (e.g., 1% means buy at -1%, -2%, -3%...)"
    )
    
    sell_profit_pct = st.number_input(
        "Sell Profit Target (%)", 
        min_value=0.1, 
        value=4.0, 
        step=0.1,
        help="Sell when price reaches X% above buy price."
    )
    
    st.divider()
    
    st.header("3. Wallet Settings")
    initial_investment = st.number_input(f"Initial Investment ({currency_symbol})", value=100000.0, step=500.0)
    shares_per_trade = st.number_input("Shares per Trade", min_value=1, value=10, step=1)
    
    run_btn = st.button("Run Backtest")

# --- Execution ---
if run_btn or 'data_loaded' in st.session_state:
    st.session_state['data_loaded'] = True
    
    # Display Strategy Summary
    st.markdown(f"""
    **Current Strategy:**
    * **Buy:** Every **{buy_drop_pct}%** drop from Previous Close.
    * **Sell:** At **{sell_profit_pct}%** profit per position.
    """)
    
    with st.spinner(f'Analyzing {ticker_symbol}...'):
        df = load_stock_data(ticker_symbol, start_input, end_input)

    if df.empty or len(df) < 2:
        st.error("No sufficient data found.")
    else:
        # --- 1. Signal Generation ---
        potential_trades = []
        for i in range(1, len(df)):
            prev_close = df.loc[i-1, 'Close']
            daily_low = df.loc[i, 'Low']
            daily_date = df.loc[i, 'Date']
            
            drop_level = 1
            while True:
                # DYNAMIC CALCULATION: drop_step * level
                current_drop_pct = buy_drop_pct * drop_level
                target_buy_price = prev_close * (1 - (current_drop_pct / 100.0))
                
                if daily_low <= target_buy_price:
                    buy_price = target_buy_price
                    # DYNAMIC CALCULATION: sell target
                    target_sell_price = buy_price * (1 + (sell_profit_pct / 100.0))
                    
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
                        "drop_pct": f"{round(current_drop_pct, 2)}%",
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
            events.append({"date": t['buy_date'], "type": "buy", "trade_id": t['trade_id'], "amount": cost})
            if t['status'] == "Closed":
                events.append({"date": t['sell_date'], "type": "sell", "trade_id": t['trade_id'], "amount": revenue})
        
        type_priority = {'sell': 0, 'buy': 1}
        events.sort(key=lambda x: (x['date'], type_priority[x['type']]))
        
        wallet = initial_investment
        active_holdings = set() 
        trade_decisions = {}
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
                    trade_cash_flows.append((event['date'], -event['amount']))
                else:
                    trade_decisions[tid] = "Missed"
                    missed_
