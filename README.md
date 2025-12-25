# Enhanced Fear & Greed Index 📈📉

**A Real-Time Stock Market Sentiment Dashboard**

This open-source web app enhances the classic **CNN Fear & Greed Index** by combining it with live retail sentiment from Reddit (r/wallstreetbets, r/stocks, r/investing) and additional free market indicators.

It delivers a faster, more retail-focused view of market psychology — helping you identify when greed or fear may be reaching extremes. Perfect for contrarian insights.

**GitHub**: https://github.com/maazkhan86/enhanced-fear-greed  
**Live Demo** (once deployed): Coming soon on Streamlit Community Cloud 🚀

## ✨ Key Features

- 📊 **Live CNN Fear & Greed Index** – the official baseline  
- 🧠 Reddit Sentiment Proxy – real-time analysis of public posts & comments (via respectful scraping, no API key required)
- ⚖️ **Custom Composite Score** – blend CNN + Reddit with an adjustable slider  
- 📅 **Historical Trends** – track scores over days/weeks with interactive charts  
- 🔍 **Subreddit Breakdown** – radar chart comparing sentiment across communities  
- ☁️ **Keyword Cloud & Top Phrases** – instantly see what’s driving the conversation  
- 📉 **Extra Indicators**  
  - VIX (volatility gauge)  
  - Equity Put/Call Ratio  
  - Google Trends for market-related searches  
- 🎯 **Stock-Specific Mode** – focus sentiment on any ticker (e.g., NVDA, TSLA)  
- ⚠️ **Smart Alerts** – clear warnings for extreme greed or fear  

All data sources are 100% free and public — no paid APIs or credentials needed.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/enhanced-fear-greed.git
   cd enhanced-fear-greed

2. **Set up a virtual environment** (recommended)
  ```env
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Activate the virtual environment**

  ```bash
  # On macOS/Linux
  source venv/bin/activate
  ```

# On Windows
venv\Scripts\activate

3. **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```
The dashboard will open in your browser automatically.

## ☁️ Free Deployment
Push your code to GitHub and deploy instantly on Streamlit Community Cloud — no cost, no server management.

## ⚖️ License
Licensed under the Apache License 2.0 — see LICENSE for details.

## 📜 Disclaimer
This tool is for educational and informational purposes only.
It is not financial advice. Sentiment indicators can be noisy or misleading, and markets are inherently unpredictable. Always perform your own research and manage risk responsibly.

## 🤝 Contributing
Contributions are welcome! Feel free to:

Open issues or pull requests
Add new data sources
Improve sentiment models
Enhance visualizations

Built with ❤️ using Python, Streamlit, BeautifulSoup, VADER, yfinance, and public data sources.
Thank you for checking out Enhanced Fear & Greed Index — stay calm while the crowd panics (or gets euphoric)! 📊
