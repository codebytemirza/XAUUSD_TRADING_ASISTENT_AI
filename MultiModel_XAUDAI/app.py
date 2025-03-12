import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
import time
import base64
from io import BytesIO
import requests
import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

class XAUUSDTradingBot:
    def __init__(self, groq_api_key, hf_api_key=None):
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="deepseek-r1-distill-llama-70b",
            api_key=groq_api_key
        )
        
        # Initialize API keys
        self.hf_api_key = hf_api_key or os.environ.get("HUGGINGFACE_API_TOKEN")
        
        # Initialize prompt templates
        self._initialize_prompts()
        
        # Initialize timeframes
        self.timeframes = {
            'D1': mt5.TIMEFRAME_D1,
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M30': mt5.TIMEFRAME_M30,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5
        }
    
    def _initialize_prompts(self):
        """Initialize all prompt templates"""
        # Main analysis prompt
        self.analysis_prompt = """Analyze XAUUSD market data for profitable trading opportunities:

Market Data:
{market_data}

Required Analysis:
1. Trend Analysis:
   - Primary trend direction across all timeframes
   - Key swing highs/lows
   - Trend line analysis

2. Price Action:
   - Support/Resistance levels
   - Chart patterns
   - Candlestick patterns
   - Market structure

3. Technical Indicators:
   - RSI divergence
   - Moving average relationships
   - ATR for volatility
   - Volume analysis

4. Key Levels:
   - Major psychological levels
   - Recent swing points
   - Fair value gaps
   - Order blocks

5. Risk Management:
   - Stop loss placement suggestions
   - Take profit targets
   - Position sizing based on risk

Provide a thorough analysis focusing on actionable trading insights."""

        # Trading signal prompt
        self.signal_prompt = """Generate a trading signal based on the following analysis:

Technical Analysis:
{technical_analysis}

Market Conditions:
{market_conditions}

Requirements:
1. Entry criteria must include:
   - Trend alignment
   - Support/Resistance respect
   - Pattern confirmation
   - Indicator confluence

2. Risk parameters:
   - Clear stop loss level
   - Multiple take profit targets
   - Maximum 1% risk per trade
   - Minimum 1:2 risk-reward ratio

3. Entry timing:
   - Specify exact entry price/zone
   - Entry trigger conditions
   - Required confirmations

Format the signal as:
DIRECTION: [BUY/SELL]
ENTRY: [Price]
STOP LOSS: [Price]
TAKE PROFIT 1: [Price]
TAKE PROFIT 2: [Price]
TAKE PROFIT 3: [Price]
TIMEFRAME: [TF]
SETUP TYPE: [Pattern/Strategy]
CONFIRMATION: [Required conditions]"""

        # Chat analysis prompt
        self.chat_prompt = """You are an expert XAUUSD trading mentor. Answer the following question based on:

Current Analysis:
{technical_analysis}

Active Signal:
{trading_signal}

User Question:
{user_question}

Provide detailed, actionable guidance focusing on both technical analysis and risk management."""

        # Initialize chains
        self.analysis_chain = PromptTemplate(
            template=self.analysis_prompt,
            input_variables=["market_data"]
        ) | self.llm

        self.signal_chain = PromptTemplate(
            template=self.signal_prompt,
            input_variables=["technical_analysis", "market_conditions"]
        ) | self.llm

        self.chat_chain = PromptTemplate(
            template=self.chat_prompt,
            input_variables=["technical_analysis", "trading_signal", "user_question"]
        ) | self.llm

    def initialize_mt5(self):
        """Initialize MetaTrader 5 connection"""
        if not mt5.initialize():
            raise Exception("MetaTrader5 initialization failed")

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        df['rsi'] = self.calculate_rsi(df)
        
        # Moving Averages
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # ATR
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
        """Get market data from MT5"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=bars)
            
            rates = mt5.copy_rates_range(
                "XAUUSDm",
                timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error getting market data: {str(e)}")
            return None

    def prepare_market_data(self):
        """Prepare market data for analysis"""
        market_data = {}
        for tf_name, tf_value in self.timeframes.items():
            df = self.get_market_data(tf_value)
            if df is not None:
                # Format the data string
                data_str = f"\n{tf_name} Timeframe Analysis:\n"
                data_str += f"Current Price: {df['close'].iloc[-1]:.2f}\n"
                data_str += f"Daily Range: {df['high'].iloc[-1]:.2f} - {df['low'].iloc[-1]:.2f}\n"
                data_str += f"RSI: {df['rsi'].iloc[-1]:.2f}\n"
                data_str += f"ATR: {df['atr'].iloc[-1]:.2f}\n"
                market_data[tf_name] = data_str
        
        return market_data

    def generate_analysis(self):
        """Generate technical analysis"""
        market_data = self.prepare_market_data()
        if not market_data:
            return None

        analysis = self.analysis_chain.invoke({
            "market_data": json.dumps(market_data, indent=2)
        })
        
        return analysis.content if hasattr(analysis, 'content') else str(analysis)

    def generate_signal(self, technical_analysis):
        """Generate trading signal"""
        market_conditions = {
            "spread": mt5.symbol_info("XAUUSDm").spread if mt5.symbol_info("XAUUSDm") else None,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        signal = self.signal_chain.invoke({
            "technical_analysis": technical_analysis,
            "market_conditions": json.dumps(market_conditions)
        })
        
        return signal.content if hasattr(signal, 'content') else str(signal)

    def chat_response(self, user_question, technical_analysis, trading_signal):
        """Generate chat response"""
        try:
            response = self.chat_chain.invoke({
                "technical_analysis": technical_analysis,
                "trading_signal": trading_signal,
                "user_question": user_question
            })
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def run_analysis(self):
        """Run complete market analysis"""
        try:
            self.initialize_mt5()
            
            # Generate analysis
            technical_analysis = self.generate_analysis()
            if not technical_analysis:
                return None
            
            # Generate signal
            trading_signal = self.generate_signal(technical_analysis)
            
            # Package results
            result = {
                "timestamp": datetime.now().isoformat(),
                "technical_analysis": technical_analysis,
                "trading_signal": trading_signal,
                "market_data": self.prepare_market_data()
            }
            
            mt5.shutdown()
            return result
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            mt5.shutdown()
            return None

def main():
    st.set_page_config(page_title="XAUUSD AI Trader", layout="wide")
    st.title("📈 XAUUSD AI Trading Assistant")
    
    # Initialize session state
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "bot" not in st.session_state:
        st.session_state.bot = None

    # API Configuration
    with st.sidebar:
        st.subheader("Configuration")
        groq_api_key = st.text_input("GROQ API Key", type="password")
        hf_api_key = st.text_input("HuggingFace API Key (Optional)", type="password")
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Analyze Market"):
            if not groq_api_key:
                st.error("Please provide GROQ API key")
            else:
                try:
                    st.session_state.bot = XAUUSDTradingBot(
                        groq_api_key=groq_api_key,
                        hf_api_key=hf_api_key if hf_api_key else None
                    )
                    with st.spinner("Analyzing market..."):
                        st.session_state.analysis = st.session_state.bot.run_analysis()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        if st.session_state.analysis:
            st.subheader("Technical Analysis")
            st.markdown(st.session_state.analysis["technical_analysis"])
            
            st.subheader("Trading Signal")
            st.markdown(st.session_state.analysis["trading_signal"])

    with col2:
        st.subheader("Chat with AI Assistant")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")
        
        # Chat input
        user_question = st.text_input("Ask about the analysis or trading setup:")
        
        if user_question and st.session_state.bot and st.session_state.analysis:
            response = st.session_state.bot.chat_response(
                user_question,
                st.session_state.analysis["technical_analysis"],
                st.session_state.analysis["trading_signal"]
            )
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Clear input
            st.experimental_rerun()
        
        elif user_question and not st.session_state.analysis:
            st.warning("Please run market analysis first")

if __name__ == "__main__":
    main()