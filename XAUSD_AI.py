import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
import time

# Enhanced Technical Analysis Feature Calculation Prompt
FEATURE_PROMPT = """You are a professional forex trader specializing in XAUUSD technical analysis.
Analyze the following XAUUSD price data across multiple timeframes to identify key technical features.

Daily Timeframe Data:
{daily_data}

4-Hour Timeframe Data:
{h4_data}

1-Hour Timeframe Data:
{h1_data}

30-Minute Timeframe Data:
{m30_data}

15-Minute Timeframe Data:
{m15_data}

5-Minute Timeframe Data:
{m5_data}

Provide a comprehensive multi-timeframe technical analysis including:
1. Overall market structure and dominant trend across timeframes
2. Key support/resistance levels from higher timeframes
3. Supply and demand zones across timeframes
4. Order blocks and institutional price levels
5. Fair Value Gaps (FVG) on lower timeframes
6. Volume analysis and liquidity zones
7. Key technical indicator readings (RSI, moving averages)
8. Scalping opportunities on 5-minute timeframe

Focus on confluence between multiple timeframes for higher probability setups.

NOTE: Prioritize higher timeframe structure while using M5 for precise entry timing.
"""

# Enhanced Trading Signal Generation Prompt
TRADING_PROMPT = """You are an institutional XAUUSD trader. Generate signals using strict order flow rules:

**Market Context**
Daily Structure: {daily_data}
4H Order Blocks: {h4_data}
1H FVG Zones: {h1_data}
30M Price Action: {m30_data}
15M Price Action: {m15_data}
5M Entry Timing: {m5_data}

**Technical Confluence**
{technical_features}

**Analysis Protocol**
1. Trend Validation:
   - Confirm alignment: Daily > 4H > 1H > 30M > 15M > 5M trends
   - Valid only if 4/5 timeframes agree
   
2. Entry Requirements:
   - Order Block + FVG confluence (±0.2% price zone)
   - Liquidity sweep of recent swing
   - BOS confirmation on executing TF
   - RSI divergence (14-period)
   - structure mapping on all Timeframes
   - 5M candlestick pattern confirmation for scalping entry
   
3. Risk Parameters:
   - Max risk: 1% equity per trade
   - Stop Loss: 1.5x ATR beyond structure
   - Minimum RR 1:2 (ideal 1:3)
   - Spread < 35 points
   - No pending high-impact news

**Execution Rules**
IF ALL CONDITIONS MET:
SIGNAL: [BUY/SELL/BUYLIMIT/SELLLIMIT]
ENTRY: [Exact price/zone]
SL: [Structural level]
TP1: [1R] | TP2: [2R] | TP3: [3R]
LOT SIZE: [Calculated 0.01-0.05 lots]

IF ANY CONDITION FAILED:
SIGNAL: NO TRADE
REASON: [Specific rule violation]

NOTE: balance 1789 cents only
"""

class XAUUSDTradingBot:
    def __init__(self, api_key):
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="deepseek-r1-distill-llama-70b",
            api_key=api_key
        )
        
        # Create feature analysis chain
        feature_prompt = PromptTemplate(
            template=FEATURE_PROMPT,
            input_variables=["daily_data", "h4_data", "h1_data", "m30_data", "m15_data", "m5_data"]
        )
        self.feature_chain = feature_prompt | self.llm
        
        # Create trading signal chain
        trading_prompt = PromptTemplate(
            template=TRADING_PROMPT,
            input_variables=["daily_data", "h4_data", "h1_data", "m30_data", "m15_data", "m5_data", "technical_features"]
        )
        self.trading_chain = trading_prompt | self.llm
        
        # Define timeframes for analysis
        self.timeframes = {
            'D1': mt5.TIMEFRAME_D1,
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M30': mt5.TIMEFRAME_M30,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5
        }
    
    def initialize_mt5(self):
        if not mt5.initialize():
            raise Exception("MetaTrader5 initialization failed")
    
    def calculate_indicators(self, df):
        """Calculate multiple technical indicators"""
        # RSI
        df['rsi'] = self.calculate_rsi(df)
        
        # Moving Averages
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # Average True Range (ATR)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def calculate_rsi(self, data, periods=14):
        """Calculate RSI indicator"""
        close_delta = data['close'].diff()
        gains = close_delta.clip(lower=0)
        losses = -1 * close_delta.clip(upper=0)
        avg_gains = gains.rolling(window=periods, min_periods=periods).mean()
        avg_losses = losses.rolling(window=periods, min_periods=periods).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_market_data(self, timeframe, bars=100):
        """Get market data for specified timeframe"""
        self.initialize_mt5()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=bars)
            
            rates = mt5.copy_rates_range(
                "XAUUSD",
                timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                print(f"No data received for timeframe {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            if df.empty:
                print("Empty dataframe received")
                return None
                
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting market data: {str(e)}")
            return None
            
        finally:
            mt5.shutdown()
    
    def prepare_data_string(self, df, timeframe_name):
        """Prepare formatted data string for analysis"""
        recent_data = df.tail(10).copy()
        data_str = f"\nRecent XAUUSD {timeframe_name} candles:\n"
        for _, row in recent_data.iterrows():
            data_str += (
                f"Time: {row['time']}, Open: {row['open']:.2f}, High: {row['high']:.2f}, "
                f"Low: {row['low']:.2f}, Close: {row['close']:.2f}, RSI: {row['rsi']:.2f}, "
                f"EMA20: {row['ema_20']:.2f}, EMA50: {row['ema_50']:.2f}, ATR: {row['atr']:.2f}\n"
            )
        return data_str
    
    def calculate_features(self, market_data):
        """Calculate technical features using LLM"""
        try:
            response = self.feature_chain.invoke({
                "daily_data": market_data['D1'],
                "h4_data": market_data['H4'],
                "h1_data": market_data['H1'],
                "m30_data": market_data['M30'],
                "m15_data": market_data['M15'],
                "m5_data": market_data['M5']
            })
            return response
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
    
    def generate_trading_signal(self, market_data, technical_features):
        """Generate trading signal using LLM"""
        try:
            response = self.trading_chain.invoke({
                "daily_data": market_data['D1'],
                "h4_data": market_data['H4'],
                "h1_data": market_data['H1'],
                "m30_data": market_data['M30'],
                "m15_data": market_data['M15'],
                "m5_data": market_data['M5'],
                "technical_features": technical_features
            })
            return response
        except Exception as e:
            print(f"Error generating signal: {e}")
            return None
    
    def run_analysis(self):
        """Main method to run the trading analysis"""
        try:
            self.initialize_mt5()
            
            # Get market data for all timeframes
            market_data = {}
            for tf_name, tf_value in self.timeframes.items():
                df = self.get_market_data(tf_value)
                if df is not None:
                    market_data[tf_name] = self.prepare_data_string(df, tf_name)
            
            # Calculate technical features
            features = self.calculate_features(market_data)
            if not features:
                return None
            
            # Generate trading signal
            signal = self.generate_trading_signal(market_data, features)
            
            # Check current spread
            symbol_info = mt5.symbol_info("XAUUSD")
            current_spread = symbol_info.spread if symbol_info else None
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "technical_features": features,
                "trading_signal": signal,
                "current_spread": current_spread
            }
            
            mt5.shutdown()
            return result
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            mt5.shutdown()
            return None

def main():
    # Initialize trading bot with your Groq API key
    api_key = "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"  # Replace with your actual Groq API key
    bot = XAUUSDTradingBot(api_key=api_key)
    
    while True:
        try:
            # Run analysis
            result = bot.run_analysis()
            
            if result:
                print("\n=== Analysis Results ===")
                print("\nCurrent Spread:", result["current_spread"])
                
                print("\nMarket Data Analysis:")
                for timeframe, data in result["market_data"].items():
                    print(f"\n{timeframe} Timeframe:")
                    print(data)
                
                print("\nTechnical Analysis:")
                print(result["technical_features"])
                
                print("\nTrading Signal:")
                print(result["trading_signal"])
                print("\n=====================")
            else:
                print("No valid analysis generated")
            
            # Wait for 30 minutes before next analysis
            print("\nWaiting 30 minutes for next analysis...")
            time.sleep(1800)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()