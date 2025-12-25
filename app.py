import streamlit as st
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from pytrends.request import TrendReq
import yfinance as yf
import re

load_dotenv()

# ============================
# Configuration & Setup
# ============================

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

SUBREDDITS = ['wallstreetbets', 'stocks', 'investing']
DEFAULT_MARKET_QUERY = "S&P 500 OR SPX OR stock market OR bull OR bear"

DATA_DIR = 'data'
HISTORICAL_FILE = os.path.join(DATA_DIR, 'historical_scores.csv')

# Create data folder and empty CSV if not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(HISTORICAL_FILE):
    pd.DataFrame(columns=['date', 'cnn_score', 'reddit_score', 'composite_score', 'vix', 'put_call_ratio']).to_csv(HISTORICAL_FILE, index=False)

# ============================
# Data Fetching Functions
# ============================

@st.cache_data(ttl=3600)  # Cache 1 hour
def get_cnn_fear_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            score = data['fear_and_greed']['score']
            rating = data['fear_and_greed']['rating']
            return round(score, 1), rating
    except:
        pass
    return None, None

@st.cache_data(ttl=1800)  # Cache 30 min for Reddit (more dynamic)
def get_reddit_sentiment(subreddits=SUBREDDITS, query=DEFAULT_MARKET_QUERY, limit=50, ticker=None):
    if ticker:
        query = f"${ticker.upper()} OR {ticker.upper()} {query}"
    
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {}
    all_text = ""

    for subreddit_name in subreddits:
        subreddit_sentiments = []
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(query, sort='new', limit=limit // len(subreddits) + 1):
                # Title + selftext
                text = submission.title + " " + (submission.selftext or "")
                all_text += text + " "
                score = analyzer.polarity_scores(text)['compound']
                subreddit_sentiments.append(score)

                # Top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:5]:
                    if comment.body:
                        all_text += comment.body + " "
                        score = analyzer.polarity_scores(comment.body)['compound']
                        subreddit_sentiments.append(score)
        except:
            pass  # Skip if subreddit access issue

        avg = sum(subreddit_sentiments) / len(subreddit_sentiments) if subreddit_sentiments else 0
        sentiments[subreddit_name] = round((avg + 1) * 50, 1)  # Normalize to 0-100

    overall_avg = sum(sentiments.values()) / len(sentiments) if sentiments else 50.0
    return round(overall_avg, 1), sentiments, all_text

def get_keyword_cloud_and_phrases(all_text):
    if not all_text.strip():
        return None, [], []

    # Clean text
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())
    words = cleaned.split()
    word_freq = Counter(words)

    # Simple bigrams for phrases
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_freq = Counter(bigrams)

    bullish = [p for p, c in bigram_freq.most_common(15) if any(w in p for w in ['bull', 'moon', 'buy', 'rally', 'gains', 'long', 'calls'])]
    bearish = [p for p, c in bigram_freq.most_common(15) if any(w in p for w in ['bear', 'crash', 'sell', 'dip', 'put', 'short', 'recession'])]

    # Word cloud
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig, bullish[:5], bearish[:5]

@st.cache_data(ttl=3600)
def get_google_trends():
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        keywords = ['stock market crash', 'buy the dip', 'recession', 'bull market']
        pytrends.build_payload(keywords, timeframe='now 7-d')
        data = pytrends.interest_over_time()
        if not data.empty:
            return round(data.mean().mean(), 1)
    except:
        pass
    return 50.0

@st.cache_data(ttl=3600)
def get_vix():
    try:
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        return round(vix, 2)
    except:
        return None

@st.cache_data(ttl=3600)
def get_put_call_ratio():
    url = "https://www.cboe.com/us/options/market_statistics/daily/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='table-daily-volume')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if cells and 'Equity' in row.text:
                    ratio = cells[-1].text.strip().replace(',', '')
                    return float(ratio) if ratio.replace('.','').isdigit() else None
    except:
        pass
    return None

# ============================
# Historical Data Handling
# ============================

def append_historical(cnn, reddit, composite, vix, put_call):
    today = datetime.now().date().isoformat()
    df = pd.read_csv(HISTORICAL_FILE)
    if today not in df['date'].values:
        new_row = {
            'date': today,
            'cnn_score': cnn,
            'reddit_score': reddit,
            'composite_score': composite,
            'vix': vix,
            'put_call_ratio': put_call
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(HISTORICAL_FILE, index=False)

def get_alert(score):
    if score >= 75:
        return "🚨 **Extreme Greed** – High risk of overbought conditions"
    elif score >= 60:
        return "📈 **Greed** – Momentum strong, caution advised"
    elif score <= 25:
        return "⚠️ **Extreme Fear** – Potential buying opportunity"
    elif score <= 40:
        return "📉 **Fear** – Market caution prevailing"
    else:
        return "🟡 **Neutral** – Balanced sentiment"

# ============================
# Streamlit App UI
# ============================

st.set_page_config(page_title="Enhanced Fear & Greed Index", layout="wide")
st.title("Enhanced Fear & Greed Index 📈📉")

st.markdown("A real-time stock market sentiment dashboard combining CNN's Fear & Greed Index with Reddit retail sentiment and multiple free indicators.")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker (optional)", placeholder="e.g., NVDA, TSLA")
with col2:
    custom_subs = st.text_input("Subreddits (comma-separated)", value=",".join(SUBREDDITS))
subs_list = [s.strip() for s in custom_subs.split(',') if s.strip()]

# Fetch Data
with st.spinner("Fetching latest data..."):
    cnn_score, cnn_rating = get_cnn_fear_greed()
    reddit_score, sub_scores, all_text = get_reddit_sentiment(subs_list, ticker=ticker)
    trends_score = get_google_trends()
    vix = get_vix()
    put_call = get_put_call_ratio()

# Composite Score
weight = st.slider("Reddit Weight in Composite Score (%)", 0, 100, 30)
composite = round((1 - weight/100) * (cnn_score or 50) + (weight/100) * reddit_score, 1)

# Save to history
append_historical(cnn_score, reddit_score, composite, vix, put_call)

# Main Metrics
st.markdown("### Core Sentiment Scores")
col1, col2, col3 = st.columns(3)
col1.metric("CNN Fear & Greed", cnn_score or "N/A", delta=cnn_rating.capitalize() if cnn_rating else None)
col2.metric("Reddit Proxy Score", reddit_score)
col3.metric("Composite Score", composite)

# Extra Indicators
st.markdown("### Market Proxies")
col1, col2, col3 = st.columns(3)
col1.metric("VIX (Volatility)", vix or "N/A")
col2.metric("Equity Put/Call Ratio", put_call or "N/A")
col3.metric("Google Trends Score", trends_score)

# Alert
st.markdown(f"### Current Market Alert")
st.markdown(f"**{get_alert(composite)}**")

# Subreddit Breakdown
st.markdown("### Subreddit Sentiment Breakdown")
sub_df = pd.DataFrame(list(sub_scores.items()), columns=['Subreddit', 'Score'])
fig_radar = go.Figure(data=go.Scatterpolar(r=sub_df['Score'], theta=sub_df['Subreddit'], fill='toself', name='Sentiment'))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
st.plotly_chart(fig_radar, use_container_width=True)

# Keyword Insights
st.markdown("### Keyword Cloud & Top Phrases")
if all_text:
    fig, bullish, bearish = get_keyword_cloud_and_phrases(all_text)
    if fig:
        st.pyplot(fig)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Bullish Phrases**")
            st.write(", ".join(bullish) if bullish else "None detected")
        with col2:
            st.markdown("**Bearish Phrases**")
            st.write(", ".join(bearish) if bearish else "None detected")
else:
    st.info("No recent Reddit text data available for keywords.")

# Historical Trends
st.markdown("### Historical Trends")
hist_df = pd.read_csv(HISTORICAL_FILE)
if len(hist_df) > 1:
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    fig_hist = px.line(hist_df, x='date', y=['cnn_score', 'reddit_score', 'composite_score'],
                       title="Sentiment Scores Over Time", markers=True)
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("Historical data will build as you refresh the app over time.")

# Refresh
if st.button("🔄 Refresh All Data"):
    st.rerun()

st.caption("Data sources: CNN, Reddit (PRAW), yfinance, CBOE, Google Trends | Not financial advice")
