import os
import re
import time
from datetime import date
from collections import Counter

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pytrends.request import TrendReq

# Optional (wordcloud) – app still runs without it
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Enhanced Fear & Greed Index", layout="wide")

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EnhancedFearGreed/1.0",
    "Accept": "application/json,text/plain,*/*",
}

CNN_HEADERS = {
    **BASE_HEADERS,
    "Referer": "https://edition.cnn.com/",
    "Origin": "https://edition.cnn.com",
}

HISTORICAL_FILE = "historical_scores.csv"

# CNN endpoint (JSON)
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# CBOE daily market stats page (HTML)
CBOE_URL = "https://www.cboe.com/us/options/market_statistics/daily/"

# Reddit public JSON (no API key) — api domain works better on hosted envs
REDDIT_NEW_JSON = "https://api.reddit.com/r/{sub}/new?limit={limit}"

# Google Trends keywords
FEAR_TERMS = ["stock market crash", "recession", "bear market", "market selloff"]
GREED_TERMS = ["buy stocks", "bull market", "stock rally", "call options"]

# Composite: require at least this many non-missing components
MIN_COMPONENTS_FOR_COMPOSITE = 2

# Basic stopwords for word/phrase extraction
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "as", "at", "by", "from",
    "i", "you", "we", "they", "he", "she", "them", "us", "my", "your", "our",
    "not", "but", "so", "if", "then", "than", "too", "very", "just", "im", "its",
    "what", "why", "how", "when", "where", "who"
}


# ----------------------------
# Helpers
# ----------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return None


def parse_tickers(ticker_input: str):
    if not ticker_input:
        return []
    parts = [p.strip().upper() for p in ticker_input.split(",")]
    return [p for p in parts if p]


def contains_any_ticker(text: str, tickers):
    if not tickers:
        return True
    if not text:
        return False
    for t in tickers:
        if re.search(rf"\b{re.escape(t)}\b", text.upper()):
            return True
    return False


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9$ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_words(texts, top_n=50):
    all_words = []
    for t in texts:
        t = clean_text(t).lower()
        words = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
        all_words.extend(words)
    return Counter(all_words).most_common(top_n)


def extract_phrases(texts, top_n=25):
    bigrams = []
    for t in texts:
        t = clean_text(t).lower()
        words = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
        for i in range(len(words) - 1):
            bigrams.append(words[i] + " " + words[i + 1])
    return Counter(bigrams).most_common(top_n)


def request_with_retry(url, headers=None, params=None, timeout=15, attempts=4, backoff=1.4):
    """
    Retry on common bot/limit status codes.
    Returns: (response_or_none, last_status_or_error_string)
    """
    headers = headers or BASE_HEADERS
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            # Retry-worthy
            if r.status_code in (429, 403, 503, 520, 521, 522):
                last = f"HTTP {r.status_code}"
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r, f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
            time.sleep(backoff * (i + 1))
    return None, last


# ----------------------------
# Data fetchers
# ----------------------------
@st.cache_data(ttl=300)
def get_cnn_fear_greed():
    """
    Returns: (score_0_100, rating_text, debug_status)
    """
    r, status = request_with_retry(
        CNN_URL,
        headers=CNN_HEADERS,
        params={"_": int(time.time())},  # cachebuster
        timeout=12,
        attempts=4,
        backoff=1.6,
    )
    if r is None:
        return None, None, status

    try:
        data = r.json()
        fag = data.get("fear_and_greed", {})
        score = safe_float(fag.get("score"))
        rating = fag.get("rating")

        if score is None:
            series = fag.get("data")
            if isinstance(series, list) and series:
                score = safe_float(series[-1].get("y"))

        if isinstance(rating, dict):
            rating = rating.get("text") or rating.get("rating")

        return score, (rating or "N/A"), status
    except Exception as e:
        return None, None, f"{status} (json parse error: {e})"


@st.cache_data(ttl=600)
def get_vix_value():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def vix_to_score(vix: float) -> float:
    if vix is None:
        return None
    # heuristic range: 10 calm -> 40 fearful
    return clamp((40.0 - vix) / (40.0 - 10.0) * 100.0)


@st.cache_data(ttl=3600)
def get_put_call_ratio_equity():
    """
    Much more reliable approach:
    - Use CBOE page with ?dt=YYYY-MM-DD
    - Regex extract "EQUITY PUT/CALL RATIO ... <number>"
    Fallback: try read_html scan.
    Returns float or None.
    """
    dt = date.today().isoformat()
    url = f"{CBOE_URL}?dt={dt}"

    r, status = request_with_retry(url, headers=BASE_HEADERS, timeout=15, attempts=4, backoff=1.5)
    if r is None:
        return None

    html = r.text

    # Primary: regex
    m = re.search(r"EQUITY\s+PUT/CALL\s+RATIO[^0-9]*([0-9]+\.[0-9]+)", html, re.I)
    if m:
        val = safe_float(m.group(1))
        if val is not None and 0.2 <= val <= 2.5:
            return float(val)

    # Fallback: read_html
    try:
        tables = pd.read_html(html)
        for df in tables:
            for _, row in df.iterrows():
                row_text = " ".join([str(x) for x in row.values]).lower()
                if "equity" in row_text and ("put/call" in row_text or "put call" in row_text or "p/c" in row_text):
                    nums = re.findall(r"\b\d+\.\d+\b", row_text)
                    for n in nums:
                        val = safe_float(n)
                        if val is not None and 0.2 <= val <= 2.5:
                            return float(val)
    except Exception:
        pass

    return None


def pcr_to_score(pcr: float) -> float:
    if pcr is None:
        return None
    # heuristic range: 0.5 greedy -> 1.3 fearful
    return clamp((1.3 - pcr) / (1.3 - 0.5) * 100.0)


@st.cache_data(ttl=1800)
def get_google_trends_score():
    """
    Returns float or None.
    pytrends often gets blocked on cloud IPs; we retry with backoff and longer timeout.
    """
    try:
        pytrends = TrendReq(
            hl="en-US",
            tz=0,
            retries=5,
            backoff_factor=0.5,
            timeout=(10, 25),
        )
        terms = FEAR_TERMS + GREED_TERMS
        pytrends.build_payload(terms, timeframe="now 7-d")
        df = pytrends.interest_over_time()

        if df is None or df.empty:
            return None

        means = df[terms].mean(axis=0)
        fear = float(means[FEAR_TERMS].mean())
        greed = float(means[GREED_TERMS].mean())

        eps = 1e-9
        net = (greed - fear) / (greed + fear + eps)  # [-1..+1] approx
        score = 50.0 + 50.0 * net
        return clamp(score)
    except Exception:
        return None


@st.cache_data(ttl=600)
def fetch_reddit_posts(sub: str, limit: int = 40):
    """
    Fetches posts from Reddit public JSON (api.reddit.com).
    Returns list of dicts with keys: title, selftext, upvotes, num_comments
    """
    url = REDDIT_NEW_JSON.format(sub=sub, limit=limit)
    r, status = request_with_retry(url, headers=BASE_HEADERS, timeout=15, attempts=4, backoff=1.6)
    if r is None:
        raise RuntimeError(status)

    data = r.json()
    children = data.get("data", {}).get("children", [])
    posts = []
    for c in children:
        d = c.get("data", {}) or {}
        posts.append({
            "title": d.get("title", "") or "",
            "selftext": d.get("selftext", "") or "",
            "upvotes": int(d.get("score") or 0),
            "num_comments": int(d.get("num_comments") or 0),
        })
    return posts


def score_reddit_subreddit(sub: str, tickers, analyzer: SentimentIntensityAnalyzer, limit: int = 40):
    """
    Returns:
      score_0_100, avg_compound, n_posts_used, texts_used(list[str]), debug_note(str|None)
    """
    try:
        posts = fetch_reddit_posts(sub, limit=limit)
    except Exception as e:
        return None, None, 0, [], f"fetch failed: {e}"

    used = []
    weights = []
    compounds = []

    for p in posts:
        text = (p.get("title", "") + " " + p.get("selftext", "")).strip()
        if not contains_any_ticker(text, tickers):
            continue
        if not text:
            continue

        comp = analyzer.polarity_scores(text)["compound"]

        up = p.get("upvotes", 0)
        cm = p.get("num_comments", 0)
        w = 1.0 + (min(up, 5000) ** 0.5) / 30.0 + (min(cm, 2000) ** 0.5) / 25.0

        compounds.append(comp)
        weights.append(w)
        used.append(text)

    if not compounds:
        note = "no ticker matches" if tickers else "no usable posts"
        return None, None, 0, [], note

    wsum = sum(weights) or 1.0
    avg_comp = sum(c * w for c, w in zip(compounds, weights)) / wsum
    score = clamp((avg_comp + 1.0) * 50.0)

    return score, avg_comp, len(compounds), used, None


def compute_composite_score(
    cnn_score,
    reddit_score,
    vix_score,
    pcr_score,
    trends_score,
    reddit_weight_pct: float
):
    """
    Composite:
    - Reddit weight is user-controlled IF Reddit exists.
    - Remaining weight split across CNN/VIX/PCR/Trends.
    - Missing dropped + renormalized.
    - If < MIN_COMPONENTS_FOR_COMPOSITE components available -> return None.
    """
    components = {}
    weights = {}

    w_reddit = (reddit_weight_pct / 100.0)

    if reddit_score is not None:
        components["Reddit"] = float(reddit_score)
        weights["Reddit"] = w_reddit

    remaining = 1.0 - weights.get("Reddit", 0.0)

    base = {
        "CNN": 0.50,
        "VIX": 0.20,
        "Put/Call": 0.20,
        "Trends": 0.10
    }

    if cnn_score is not None:
        components["CNN"] = float(cnn_score)
        weights["CNN"] = remaining * base["CNN"]

    if vix_score is not None:
        components["VIX"] = float(vix_score)
        weights["VIX"] = remaining * base["VIX"]

    if pcr_score is not None:
        components["Put/Call"] = float(pcr_score)
        weights["Put/Call"] = remaining * base["Put/Call"]

    if trends_score is not None:
        components["Trends"] = float(trends_score)
        weights["Trends"] = remaining * base["Trends"]

    # Require at least 2 components, otherwise composite is misleading
    if len(components) < MIN_COMPONENTS_FOR_COMPOSITE:
        return None, {}, {}, "Insufficient data: need at least 2 components."

    total_w = sum(weights.values())
    if total_w <= 0:
        return None, {}, {}, "Composite weights sum to 0."

    composite = sum(components[k] * weights[k] for k in components) / total_w
    composite = round(float(composite), 1)

    weights_norm = {k: (weights[k] / total_w) for k in weights}
    return composite, weights_norm, components, None


def market_label(score: float) -> str:
    if score is None:
        return "Unknown"
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def save_history_row(row: dict):
    today_str = date.today().isoformat()
    df_new = pd.DataFrame([row])
    df_new["date"] = today_str

    if os.path.exists(HISTORICAL_FILE):
        try:
            df = pd.read_csv(HISTORICAL_FILE)
            if "date" in df.columns and (df["date"] == today_str).any():
                df.loc[df["date"] == today_str, df_new.columns] = df_new.iloc[0].values
            else:
                df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(HISTORICAL_FILE, index=False)
            return
        except Exception:
            pass

    df_new.to_csv(HISTORICAL_FILE, index=False)


# ----------------------------
# UI
# ----------------------------
st.title("Enhanced Fear & Greed Index 📈📉")
st.caption(
    "Real-time sentiment dashboard blending CNN Fear & Greed, Reddit sentiment (public JSON), VIX, equity put/call ratio, and Google Trends."
)

colA, colB = st.columns([2, 1])
with colA:
    ticker_input = st.text_input(
        "Stock Ticker(s) (optional – filters Reddit titles/body)",
        placeholder="e.g., NVDA, TSLA"
    )
with colB:
    subs_input = st.text_input("Subreddits (comma-separated)", value="wallstreetbets,stocks,investing")

reddit_weight = st.slider(
    "Reddit Weight in Composite Score (%)",
    min_value=0, max_value=60, value=30, step=5
)

tickers = parse_tickers(ticker_input)
subreddits = [s.strip() for s in subs_input.split(",") if s.strip()]
subreddits = list(dict.fromkeys(subreddits))  # de-dupe, preserve order

analyzer = SentimentIntensityAnalyzer()

# ----------------------------
# Fetch + compute
# ----------------------------
with st.spinner("Fetching data…"):
    cnn_score, cnn_rating, cnn_debug = get_cnn_fear_greed()

    # Reddit per-sub scoring
    subreddit_rows = []
    all_texts = []
    reddit_scores = []
    reddit_debug_notes = {}

    for sub in subreddits:
        time.sleep(0.15)
        s_score, s_comp, n_used, texts_used, note = score_reddit_subreddit(sub, tickers, analyzer, limit=50)
        if texts_used:
            all_texts.extend(texts_used)
        if s_score is not None:
            reddit_scores.append(s_score)
        if note:
            reddit_debug_notes[sub] = note

        subreddit_rows.append({
            "subreddit": sub,
            "score_0_100": None if s_score is None else round(float(s_score), 1),
            "avg_compound": None if s_comp is None else round(float(s_comp), 3),
            "posts_used": int(n_used),
        })

    reddit_score = round(float(sum(reddit_scores) / len(reddit_scores)), 1) if reddit_scores else None

    # Proxies
    vix_val = get_vix_value()
    vix_score = None if vix_val is None else round(vix_to_score(vix_val), 1)

    pcr_val = get_put_call_ratio_equity()
    pcr_score = None if pcr_val is None else round(pcr_to_score(pcr_val), 1)

    trends_score = get_google_trends_score()
    trends_score = None if trends_score is None else round(float(trends_score), 1)

    composite, weights_used, comps_used, composite_note = compute_composite_score(
        cnn_score=cnn_score,
        reddit_score=reddit_score,
        vix_score=vix_score,
        pcr_score=pcr_score,
        trends_score=trends_score,
        reddit_weight_pct=reddit_weight
    )

# ----------------------------
# Display: Core scores
# ----------------------------
st.markdown("## Core Sentiment Scores")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("CNN Fear & Greed", "N/A" if cnn_score is None else f"{cnn_score:.1f}",
              delta=cnn_rating if cnn_rating else None)
    st.caption("Source: CNN (graphdata JSON)")

with c2:
    if reddit_score is None:
        st.metric("Reddit Proxy (Retail Sentiment)", "N/A")
        if tickers:
            st.caption("No matching ticker mentions found in recent posts (or fetch blocked).")
        else:
            st.caption("Reddit fetch failed or no usable posts.")
    else:
        st.metric("Reddit Proxy (Retail Sentiment)", f"{reddit_score:.1f}")
        st.caption("Source: Reddit public JSON + VADER (weighted by engagement)")

with c3:
    st.metric("Composite Score (Enhanced)", "N/A" if composite is None else f"{composite:.1f}")
    if composite is None and composite_note:
        st.caption(f"Regime: Unknown • {composite_note}")
    else:
        st.caption(f"Regime: {market_label(composite)}" if composite is not None else "Regime: Unknown")

# Explain weights actually used
with st.expander("How the composite is built (weights used for this run)"):
    if not weights_used:
        st.write("No composite available (missing inputs).")
    else:
        wdf = pd.DataFrame(
            [{"Component": k, "Weight": round(v, 3), "Score": round(comps_used.get(k, float("nan")), 1)}
             for k, v in weights_used.items()]
        ).sort_values("Weight", ascending=False)
        st.dataframe(wdf, use_container_width=True)
        st.caption("If a source fails (e.g., Put/Call N/A), it is dropped and the remaining weights are renormalized.")

# ----------------------------
# Display: Market proxies
# ----------------------------
st.markdown("## Market Proxies")
p1, p2, p3 = st.columns(3)

with p1:
    st.metric("VIX (raw)", "N/A" if vix_val is None else f"{vix_val:.2f}")
    st.caption("Lower VIX generally = more greed / calm markets")
    st.metric("VIX → score", "N/A" if vix_score is None else f"{vix_score:.1f}")

with p2:
    st.metric("Equity Put/Call Ratio (raw)", "N/A" if pcr_val is None else f"{pcr_val:.3f}")
    st.caption("Higher Put/Call generally = more fear / hedging")
    st.metric("Put/Call → score", "N/A" if pcr_score is None else f"{pcr_score:.1f}")

with p3:
    st.metric("Google Trends → score", "N/A" if trends_score is None else f"{trends_score:.1f}")
    st.caption("Net signal: greed-search terms vs fear-search terms (last 7 days)")

# Alert
st.markdown(f"### Market Alert: **{market_label(composite)}**")

# ----------------------------
# Debug expander (helps you see cloud blocks)
# ----------------------------
with st.expander("Debug: data source connectivity / reasons for N/A"):
    st.write("CNN fetch:", cnn_debug)

    # Quick status probes (non-cached)
    _, reddit_probe = request_with_retry("https://api.reddit.com/r/wallstreetbets/new?limit=1", attempts=1)
    st.write("Reddit probe:", reddit_probe)

    _, cboe_probe = request_with_retry(f"{CBOE_URL}?dt={date.today().isoformat()}", attempts=1)
    st.write("CBOE probe:", cboe_probe)

    st.write("Notes per subreddit (if any):", reddit_debug_notes if reddit_debug_notes else "—")

    st.write("If you see HTTP 403/418/429, the provider is blocking Streamlit Cloud IPs.")
    st.write("Composite is intentionally disabled unless at least 2 components are available.")

# ----------------------------
# Subreddit breakdown
# ----------------------------
st.markdown("## Subreddit Breakdown")
sub_df = pd.DataFrame(subreddit_rows)
st.dataframe(sub_df, use_container_width=True)

chart_df = sub_df.dropna(subset=["score_0_100"])
if not chart_df.empty:
    fig = px.bar(chart_df, x="subreddit", y="score_0_100", title="Reddit sentiment score by subreddit (0–100)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Words / phrases / wordcloud
# ----------------------------
st.markdown("## Crowd Language (Reddit)")

if all_texts:
    words = extract_words(all_texts, top_n=40)
    phrases = extract_phrases(all_texts, top_n=25)

    wcol, pcol = st.columns(2)
    with wcol:
        st.markdown("**Top words**")
        st.write(pd.DataFrame(words, columns=["word", "count"]))
    with pcol:
        st.markdown("**Top phrases**")
        st.write(pd.DataFrame(phrases, columns=["phrase", "count"]))

    if WORDCLOUD_AVAILABLE:
        st.markdown("**Word Cloud**")
        wc_text = " ".join(clean_text(t) for t in all_texts)
        wc = WordCloud(width=1200, height=450, background_color="white").generate(wc_text)
        fig_wc = plt.figure(figsize=(12, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig_wc, clear_figure=True)
else:
    st.info("No Reddit text available to build phrases/word cloud (try removing ticker filter or changing subreddits).")

# ----------------------------
# Historical tracking
# ----------------------------
st.markdown("## Historical Trends")

history_row = {
    "cnn_score": None if cnn_score is None else round(float(cnn_score), 1),
    "reddit_score": None if reddit_score is None else round(float(reddit_score), 1),
    "vix_raw": None if vix_val is None else round(float(vix_val), 3),
    "vix_score": None if vix_score is None else round(float(vix_score), 1),
    "putcall_raw": None if pcr_val is None else round(float(pcr_val), 4),
    "putcall_score": None if pcr_score is None else round(float(pcr_score), 1),
    "trends_score": None if trends_score is None else round(float(trends_score), 1),
    "composite_score": None if composite is None else round(float(composite), 1),
}

try:
    save_history_row(history_row)
except Exception:
    pass

if os.path.exists(HISTORICAL_FILE):
    try:
        hist_df = pd.read_csv(HISTORICAL_FILE)
        if len(hist_df) > 1:
            hist_df["date"] = pd.to_datetime(hist_df["date"])
            fig_hist = px.line(
                hist_df.sort_values("date"),
                x="date",
                y=["cnn_score", "reddit_score", "vix_score", "putcall_score", "trends_score", "composite_score"],
                markers=True,
                title="Daily component scores (0–100) + composite"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Run the app daily to build a useful history series.")
    except Exception:
        st.warning("Could not read historical file (format issue).")

# Controls
c_refresh, c_clear = st.columns([1, 1])
with c_refresh:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with c_clear:
    if st.button("🧹 Clear history file (danger)"):
        try:
            if os.path.exists(HISTORICAL_FILE):
                os.remove(HISTORICAL_FILE)
            st.success("History cleared.")
            st.rerun()
        except Exception:
            st.error("Could not clear history file.")

st.caption(
    "Sources: CNN (graphdata JSON), Reddit public JSON, yfinance (^VIX), CBOE daily stats (HTML), Google Trends (pytrends) | Not financial advice"
)
