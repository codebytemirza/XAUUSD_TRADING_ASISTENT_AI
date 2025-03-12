# XAUUSD AI Trading Bot 🤖

An advanced algorithmic trading system for XAUUSD (Gold) using multi-timeframe analysis and AI-powered decision making. 65% of accuracy for profitable trades tested on real account

## Features 🌟

- Multi-timeframe Technical Analysis (D1, H4, H1, M30, M15, M5)
- AI-powered trade signal generation using Groq LLM
- Real-time market data analysis via MetaTrader 5
- Comprehensive technical indicators (RSI, EMA, ATR)
- Interactive Streamlit web dashboard
- Automated trade suggestion system
- Risk management protocols

## Prerequisites 📋

- Python 3.8+
- MetaTrader 5 with active account
- Groq API key
- Required Python packages:
    ```
    MetaTrader5
    pandas
    numpy
    langchain-groq
    streamlit
    ```

## Installation 🔧

1. Clone the repository:
```bash
git clone <repository-url>
cd <repo folder name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your-groq-api-key"
```

## Usage 🚀

1. Start the Streamlit dashboard:
```bash
streamlit run app.py
```

2. Access features through the web interface:
- View multi-timeframe analysis
- Get AI-generated trading signals
- Monitor market conditions
- Track spread changes
- Auto-refresh market analysis

## Project Structure 📁

```
CourseBot/
│
├── XAUSD_AI.py      # Core trading bot logic
├── app.py           # Streamlit dashboard
├── requirements.txt # Dependencies
└── README.md       # Documentation
```

## Trading Features 📊

- Order block identification
- Fair Value Gap (FVG) analysis
- Supply and demand zones
- Risk calculation (1% per trade)
- Dynamic stop-loss using ATR
- Multi-timeframe confluence

## Disclaimer ⚠️

This bot is for educational purposes only. Always verify signals and manage risk appropriately. Trading involves substantial risk of loss.

## License 📝

MIT License - See LICENSE file for details
