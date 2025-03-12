import streamlit as st
from datetime import datetime
from XAUSD_AI import XAUUSDTradingBot
import time

# Must be the first Streamlit command
st.set_page_config(page_title="XAUUSD Trading Bot", page_icon="📈", layout="wide")

# Initialize trading bot with API key from secrets
bot = XAUUSDTradingBot(api_key=st.secrets["GROQ_API_KEY"])

def display_market_data(data_str, timeframe):
    """Display formatted market data in an expander"""
    with st.expander(f"📊 {timeframe} Market Data", expanded=False):
        lines = data_str.split('\n')
        for line in lines:
            if line.strip():
                st.text(line)

def format_signal(signal_content):
    """Format and display trading signal content"""
    if isinstance(signal_content, str):
        return signal_content
    return signal_content.content if hasattr(signal_content, 'content') else str(signal_content)

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {background-color: #0e1117}
        .metric-container {
            background-color: #1f2937;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #1f2937;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #374151;
            padding: 10px 20px;
            border-radius: 5px 5px 0 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🤖 XAUUSD Trading Bot Dashboard")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Control Panel")
        auto_refresh = st.toggle("🔄 Auto Refresh (30min)", value=False)
        
        if st.button("📈 Run New Analysis"):
            with st.spinner("Analyzing market data..."):
                st.session_state['analysis_result'] = bot.run_analysis()
                st.session_state['last_update'] = datetime.now()

    # Main content area
    try:
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["📊 Analysis", "🎯 Trading Signal"])

        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            
            # Display current spread
            if result.get('current_spread'):
                st.sidebar.metric("Current Spread", f"{result['current_spread']} points")
            
            with tab1:
                # Technical Analysis
                st.markdown("### 📊 Technical Analysis")
                st.markdown(format_signal(result['technical_features']))
                
                # Market Data
                st.markdown("### 📈 Market Data by Timeframe")
                cols = st.columns(2)
                for idx, (tf, data) in enumerate(result['market_data'].items()):
                    with cols[idx % 2]:
                        display_market_data(data, tf)
            
            with tab2:
                # Trading Signal
                st.markdown("### 🎯 Trading Signal")
                st.markdown(format_signal(result['trading_signal']))
            
            # Show last update time
            st.sidebar.info(f"Last Updated: {st.session_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("⚠️ No analysis results yet. Click 'Run Analysis' to start.")
        
        # Auto-refresh progress bar
        if auto_refresh:
            current_time = datetime.now()
            if 'last_refresh' not in st.session_state:
                st.session_state['last_refresh'] = current_time
            
            time_diff = (current_time - st.session_state['last_refresh']).total_seconds()
            remaining_time = 1800 - time_diff
            
            if remaining_time > 0:
                progress = (1800 - remaining_time) / 1800
                mins = int(remaining_time // 60)
                secs = int(remaining_time % 60)
                
                st.sidebar.progress(progress, text=f"Next refresh in: {mins:02d}:{secs:02d}")
            
            if time_diff >= 1800:
                st.session_state['last_refresh'] = current_time
                time.sleep(1)
                st.rerun()
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
