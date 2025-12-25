import streamlit as st
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from pytrends.request import TrendReq
import yfinance as yf
import re
import time  # For polite delays between requests

# ============================
# Configuration
# ============================

SUBREDDITS = ['wallstreetbets', 'stocks', 'investing']
DEFAULT_MARKET_QUERY = "S&P 500 OR SPX OR stock market OR bull OR bear"  # Not used directly in scraping, but kept for future

DATA_DIR = 'data'
HISTORICAL_FILE = os.path.join(DATA_DIR, 'historical_scores.csv')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(HISTORICAL_FILE):
    pd.DataFrame(columns=['date', 'cnn_score', 'reddit_score', 'composite_score', 'vix', 'put_call_ratio']).to_csv(HISTORICAL_FILE, index=False)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
}

# ============================
# Data Fetching Functions
# ============================

@st.cache_data(ttl=3600)
def get_cnn_fear_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            score = data['fear_and_greed']['score']
            rating = data['fear_and_greed']['rating']
            return round(score, 1), rating
    except:
        pass
    return None, None

@st.cache_data(ttl=1800)  # Cache 30 minutes
def get_reddit_sentiment_scrape(subreddits=SUBREDDITS, limit_per_sub=15, ticker=None):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {}
    all_text = ""

    for subreddit_name in subreddits:
        subreddit_sentiments = []
        url = f"https://www.reddit.com/r/{subreddit_name}/new/"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            posts = soup.find_all('shreddit-post')[:limit_per_sub]  # Modern Reddit uses <shreddit-post>

            if not posts:  # Fallback for old layout
                posts = soup.find_all('div', {'data-testid': 'post-container'})[:limit_per_sub]

            for post in posts:
                # Title
                title_tag = post.find('h3') or post.find('a', slot='title')
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    if ticker and ticker.upper() not in title.upper():
                        continue  # Skip if ticker specified and not mentioned
                    all_text += title + " "
                    score = analyzer.polarity_scores(title)['compound']
                    subreddit_sentiments.append(score)

                # Selftext / body
                body = post.find('div', slot='text-body')
                if body:
                    text = body.get_text(strip=True)
                    all_text += text + " "
                    score = analyzer.polarity_scores(text)['compound']
                    subreddit_sentiments.append(score)

                # Comments (top 3–5 if available)
                comment_section = post.find('shreddit-comments')
                if comment_section:
                    comments = comment_section.find_all('div', slot='comment')[:5]
                    for comment in comments:
                        comment_text = comment.get_text(strip=True)
                        if comment_text:
                            all_text += comment_text + " "
                            score = analyzer.polarity_scores(comment_text)['compound']
                            subreddit_sentiments.append(score)

            time.sleep(1)  # Be respectful to Reddit servers

        except Exception as e:
            st.warning(f"Error scraping r/{subreddit_name}: {str(e)}")
            continue

        avg = sum(subreddit_sentiments) / len(subreddit_sentiments) if subreddit_sentiments else 0
        sentiments[subreddit_name] = round((avg + 1) * 50, 1)

    overall_avg = sum(sentiments.values()) / len(sentiments) if sentiments else 50.0
    return round(overall_avg, 1), sentiments, all_text

# Reuse the same keyword, trends, vix, put/call functions as before
def get_keyword_cloud_and_phrases(all_text):
    if not all_text.strip():
        return None, [], []

    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())
    words = cleaned.split()
    word_freq = Counter(words)

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_freq = Counter(bigrams)

    bullish = [p for p, c in bigram_freq.most_common(15) if any(w in p for w in ['bull', 'moon', 'buy', 'rally', 'gains', 'long', 'calls'])]
    bearish = [p for p, c in bigram_freq.most_common(15) if any(w in p for w in ['bear', 'crash', 'sell', 'dip', 'put', 'short', 'recession'])]

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
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='table-daily-volume')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if cells and 'Equity' in row.text:
                    ratio_text = cells[-1].text.strip().replace(',', '')
                    return round(float(ratio_text), 3) if ratio_text.replace('.','').isdigit() else None
    except:
        pass
    return None

# Historical & Alert (same as before)
def append_historical(cnn, reddit, composite, vix, put_call):
    today = datetime.now().date().isoformat()
    df = pd.read_csv(HISTORICAL_FILE)
    if today not in df['date'].values:
        new_row = {'date': today, 'cnn_score': cnn, 'reddit_score': reddit, 'composite_score': composite, 'vix': vix, 'put_call_ratio': put_call}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(HISTORICAL_FILE, index=False)

def get_alert(score):
    if score >= 75: return "🚨 **Extreme Greed** – High risk of overbought conditions"
    elif score >= 60: return "📈 **Greed** – Momentum strong, caution advised"
    elif score <= 25: return "⚠️ **Extreme Fear** – Potential buying opportunity"
    elif score <= 40: return "📉 **Fear** – Market caution prevailing"
    else: return "🟡 **Neutral** – Balanced sentiment"

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="Enhanced Fear & Greed Index", layout="wide")
st.title("Enhanced Fear & Greed Index 📈📉")
st.markdown("Real-time sentiment dashboard using **public Reddit scraping** (no API key needed), CNN index, VIX, put/call ratio, and Google Trends.")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker (optional – filters titles)", placeholder="e.g., NVDA, TSLA")
with col2:
    custom_subs = st.text_input("Subreddits (comma-separated)", value=",".join(SUBREDDITS))
subs_list = [s.strip() for s in custom_subs.split(',') if s.strip()]

with st.spinner("Scraping Reddit and fetching market data..."):
    cnn_score, cnn_rating = get_cnn_fear_greed()
    reddit_score, sub_scores, all_text = get_reddit_sentiment_scrape(subs_list, ticker=ticker if ticker else None)
    trends_score = get_google_trends()
    vix = get_vix()
    put_call = get_put_call_ratio()

weight = st.slider("Reddit Weight in Composite Score (%)", 0, 100, 30)
composite = round((1 - weight/100) * (cnn_score or 50) + (weight/100) * reddit_score, 1)

append_historical(cnn_score, reddit_score, composite, vix, put_call)

# Dashboard
st.markdown("### Core Sentiment Scores")
col1, col2, col3 = st.columns(3)
col1.metric("CNN Fear & Greed", cnn_score or "N/A", delta=cnn_rating.capitalize() if cnn_rating else None)
col2.metric("Reddit Proxy (Scraped)", reddit_score)
col3.metric("Composite Score", composite)

st.markdown("### Market Proxies")
col1, col2, col3 = st.columns(3)
col1.metric("VIX", vix or "N/A")
col2.metric("Put/Call Ratio", put_call or "N/A")
col3.metric("Google Trends", trends_score)

st.markdown(f"### Market Alert: **{get_alert(composite)}**")

st.markdown("### Subreddit Breakdown")
sub_df = pd.DataFrame(list(sub_scores.items()), columns=['Subreddit', 'Score'])
fig_radar = go.Figure(data=go.Scatterpolar(r=sub_df['Score'], theta=sub_df['Subreddit'], fill='toself'))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("### Keyword Cloud & Phrases")
if all_text:
    fig, bullish, bearish = get_keyword_cloud_and_phrases(all_text)
    if fig:
        st.pyplot(fig)
        col1, col2 = st.columns(2)
        with col1: st.markdown("**Bullish Phrases**"); st.write(", ".join(bullish) or "None")
        with col2: st.markdown("**Bearish Phrases**"); st.write(", ".join(bearish) or "None")
else:
    st.info("No text scraped yet.")

st.markdown("### Historical Trends")
hist_df = pd.read_csv(HISTORICAL_FILE)
if len(hist_df) > 1:
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    fig_hist = px.line(hist_df, x='date', y=['cnn_score', 'reddit_score', 'composite_score'], markers=True)
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("Run the app daily to build history.")

if st.button("🔄 Refresh Data"):
    st.rerun()

st.caption("Sources: CNN, public Reddit (scraped), yfinance, CBOE, Google Trends | Not financial advice")
