import os
import re
import json
import time
from datetime import date, datetime, timezone
from collections import Counter

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pytrends.request import TrendReq

# Optional wordcloud
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# Optional BeautifulSoup (recommended for CBOE parsing robustness)
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False


# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Enhanced Fear & Greed Index", layout="wide")

HISTORICAL_FILE = "historical_scores.csv"
LAST_KNOWN_FILE = "last_known.json"

DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

MIN_COMPONENTS_FOR_COMPOSITE = 2

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CBOE_URL = "https://www.cboe.com/us/options/market_statistics/daily/"

# Two endpoints (cloud environments sometimes block one but not the other)
REDDIT_ENDPOINTS = [
    "https://api.reddit.com/r/{sub}/new?limit={limit}",
    "https://www.reddit.com/r/{sub}/new.json?limit={limit}",
]

FEAR_TERMS = ["stock market crash", "recession", "bear market", "market selloff"]
GREED_TERMS = ["buy stocks", "bull market", "stock rally", "call options"]

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EnhancedFearGreed/1.2",
    "Accept": "application/json,text/plain,*/*",
}
CNN_HEADERS = {
    **BASE_HEADERS,
    "Referer": "https://edition.cnn.com/",
    "Origin": "https://edition.cnn.com",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "as", "at", "by", "from",
    "i", "you", "we", "they", "he", "she", "them", "us", "my", "your", "our",
    "not", "but", "so", "if", "then", "than", "too", "very", "just", "im", "its",
    "what", "why", "how", "when", "where", "who"
}


# ----------------------------
# Last-known storage (prevents empty dashboard)
# ----------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def load_last_known():
    try:
        if os.path.exists(LAST_KNOWN_FILE):
            with open(LAST_KNOWN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_last_known(d):
    try:
        with open(LAST_KNOWN_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass

def get_last_known_value(store, key, max_age_hours=72):
    """
    Returns (value, used_last_known_bool)
    """
    rec = store.get(key)
    if not rec:
        return None, False
    ts = rec.get("updated_at")
    val = rec.get("value")
    if val is None or ts is None:
        return None, False

    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
        if age_hours <= max_age_hours:
            return val, True
    except Exception:
        return None, False

    return None, False

def set_last_known_value(store, key, value):
    if value is None:
        return
    store[key] = {"value": value, "updated_at": utc_now_iso()}


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

def request_with_retry(url, headers=None, params=None, timeout=15, attempts=4, backoff=1.5):
    """
    Retry for rate-limits / transient blocks.
    Returns (response_or_none, debug_string)
    """
    headers = headers or BASE_HEADERS
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (429, 403, 418, 503, 520, 521, 522):
                last = f"HTTP {r.status_code}"
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r, f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
            time.sleep(backoff * (i + 1))
    return None, last

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

def confidence_label(available: int, total: int) -> str:
    if total <= 0:
        return "Unknown"
    if available >= 5:
        return "High"
    if available >= 3:
        return "Medium"
    if available >= 2:
        return "Low"
    return "Unknown"

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9$ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_words(texts, top_n=40):
    words = []
    for t in texts:
        t = clean_text(t).lower()
        ws = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
        words.extend(ws)
    return Counter(words).most_common(top_n)

def extract_phrases(texts, top_n=25):
    bigrams = []
    for t in texts:
        t = clean_text(t).lower()
        ws = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
        for i in range(len(ws) - 1):
            bigrams.append(ws[i] + " " + ws[i + 1])
    return Counter(bigrams).most_common(top_n)


# ----------------------------
# Fetchers
# ----------------------------
@st.cache_data(ttl=300)
def get_cnn_fear_greed():
    r, debug = request_with_retry(
        CNN_URL,
        headers=CNN_HEADERS,
        params={"_": int(time.time())},
        timeout=12,
        attempts=4,
        backoff=1.6,
    )
    if r is None:
        return None, None, debug
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

        return score, (rating or "N/A"), debug
    except Exception as e:
        return None, None, f"{debug} (parse error: {e})"


@st.cache_data(ttl=600)
def get_vix_value_yfinance():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=600)
def get_vix_value_stooq():
    """
    Fallback if yfinance fails: Stooq CSV endpoint.
    """
    try:
        url = "https://stooq.com/q/l/?s=vix&i=d"
        r, _ = request_with_retry(url, headers=BASE_HEADERS, timeout=12, attempts=3, backoff=1.4)
        if r is None:
            return None
        df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd, "compat") else pd.read_csv(pd.io.common.StringIO(r.text))
        # Some pandas builds differ; safer:
    except Exception:
        try:
            import io
            df = pd.read_csv(io.StringIO(r.text))
        except Exception:
            return None

    try:
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def vix_to_score(vix: float) -> float:
    if vix is None:
        return None
    return clamp((40.0 - vix) / (40.0 - 10.0) * 100.0)


def extract_equity_pcr(html: str):
    """
    Primary: regex on raw HTML.
    Fallback: BeautifulSoup text + regex (more resilient to markup/spacing).
    """
    m = re.search(r"EQUITY\s+PUT/CALL\s+RATIO[^0-9]*([0-9]+\.[0-9]+)", html, re.I)
    if m:
        return safe_float(m.group(1))

    if BS4_AVAILABLE:
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = " ".join(soup.get_text(" ").split())
            m2 = re.search(r"EQUITY\s+PUT/CALL\s+RATIO[^0-9]*([0-9]+\.[0-9]+)", text, re.I)
            if m2:
                return safe_float(m2.group(1))
        except Exception:
            return None

    return None


@st.cache_data(ttl=3600)
def get_put_call_ratio_equity():
    dt = date.today().isoformat()
    url = f"{CBOE_URL}?dt={dt}"
    r, _ = request_with_retry(url, headers=BASE_HEADERS, timeout=15, attempts=4, backoff=1.5)
    if r is None:
        return None

    val = extract_equity_pcr(r.text)
    if val is not None and 0.2 <= val <= 2.5:
        return float(val)
    return None


def pcr_to_score(pcr: float) -> float:
    if pcr is None:
        return None
    return clamp((1.3 - pcr) / (1.3 - 0.5) * 100.0)


@st.cache_data(ttl=1800)
def get_google_trends_score():
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
        net = (greed - fear) / (greed + fear + eps)
        score = 50.0 + 50.0 * net
        return clamp(score)
    except Exception:
        return None


@st.cache_data(ttl=600)
def fetch_reddit_posts(sub: str, limit: int = 60):
    """
    Tries multiple Reddit endpoints.
    """
    last_err = None
    for tmpl in REDDIT_ENDPOINTS:
        url = tmpl.format(sub=sub, limit=limit)
        r, debug = request_with_retry(url, headers=BASE_HEADERS, timeout=15, attempts=3, backoff=1.6)
        if r is None:
            last_err = debug
            continue

        try:
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
            if posts:
                return posts
            last_err = "no posts"
        except Exception as e:
            last_err = f"parse error: {e}"
            continue

    raise RuntimeError(last_err or "Reddit unavailable")


def score_reddit_subreddit(sub: str, analyzer: SentimentIntensityAnalyzer):
    try:
        posts = fetch_reddit_posts(sub, limit=60)
    except Exception as e:
        return None, None, 0, [], f"unavailable ({e})"

    compounds, weights, used = [], [], []
    for p in posts:
        text = (p.get("title", "") + " " + p.get("selftext", "")).strip()
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
        return None, None, 0, [], "no usable posts"

    wsum = sum(weights) or 1.0
    avg_comp = sum(c * w for c, w in zip(compounds, weights)) / wsum
    score = clamp((avg_comp + 1.0) * 50.0)
    return score, avg_comp, len(compounds), used, None


def compute_composite_score(cnn_score, reddit_score, vix_score, pcr_score, trends_score, reddit_weight_pct: float):
    components = {}
    weights = {}

    w_reddit = reddit_weight_pct / 100.0
    if reddit_score is not None:
        components["Reddit mood"] = float(reddit_score)
        weights["Reddit mood"] = w_reddit

    remaining = 1.0 - weights.get("Reddit mood", 0.0)

    base = {
        "CNN": 0.50,
        "VIX calmness": 0.20,
        "Options hedging": 0.20,
        "Search interest": 0.10,
    }

    if cnn_score is not None:
        components["CNN"] = float(cnn_score)
        weights["CNN"] = remaining * base["CNN"]

    if vix_score is not None:
        components["VIX calmness"] = float(vix_score)
        weights["VIX calmness"] = remaining * base["VIX calmness"]

    if pcr_score is not None:
        components["Options hedging"] = float(pcr_score)
        weights["Options hedging"] = remaining * base["Options hedging"]

    if trends_score is not None:
        components["Search interest"] = float(trends_score)
        weights["Search interest"] = remaining * base["Search interest"]

    if len(components) < MIN_COMPONENTS_FOR_COMPOSITE:
        return None, {}, {}, "Not enough signals available to calculate a reliable composite."

    total_w = sum(weights.values())
    if total_w <= 0:
        return None, {}, {}, "Weights sum to zero."

    composite = sum(components[k] * weights[k] for k in components) / total_w
    composite = round(float(composite), 1)

    weights_norm = {k: weights[k] / total_w for k in weights}
    return composite, weights_norm, components, None


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
st.caption("A simple snapshot of overall market mood using multiple public signals.")

with st.form("run_form"):
    st.markdown("### Run the snapshot")
    st.write("Click the button to pull the latest signals and calculate today’s market mood.")
    reddit_weight = st.slider(
        "How much should Reddit mood influence the final score?",
        min_value=0, max_value=60, value=30, step=5
    )
    submitted = st.form_submit_button("▶ Run analysis")

if not submitted:
    st.info("Ready when you are. Click **Run analysis** to generate today’s snapshot.")
    st.caption("Data sources: CNN, Reddit, CBOE, Google Trends, Yahoo Finance • Educational only")
    st.stop()

store = load_last_known()
analyzer = SentimentIntensityAnalyzer()

with st.spinner("Running analysis…"):
    # CNN
    cnn_score, cnn_rating, _ = get_cnn_fear_greed()
    if cnn_score is not None:
        set_last_known_value(store, "cnn_score", cnn_score)

    # Reddit
    subreddit_rows = []
    all_texts = []
    reddit_scores = []

    for sub in DEFAULT_SUBREDDITS:
        time.sleep(0.15)
        s_score, s_comp, n_used, texts_used, _ = score_reddit_subreddit(sub, analyzer)
        if texts_used:
            all_texts.extend(texts_used)
        if s_score is not None:
            reddit_scores.append(s_score)

        subreddit_rows.append({
            "subreddit": sub,
            "score_0_100": None if s_score is None else round(float(s_score), 1),
            "avg_compound": None if s_comp is None else round(float(s_comp), 3),
            "posts_used": int(n_used),
        })

    reddit_score = round(float(sum(reddit_scores) / len(reddit_scores)), 1) if reddit_scores else None
    if reddit_score is not None:
        set_last_known_value(store, "reddit_score", reddit_score)

    # VIX (yfinance -> stooq -> last-known)
    vix_val = get_vix_value_yfinance()
    used_last_vix = False
    if vix_val is None:
        vix_val = get_vix_value_stooq()
    if vix_val is None:
        vix_val, used_last_vix = get_last_known_value(store, "vix_raw", max_age_hours=72)
    if vix_val is not None and not used_last_vix:
        set_last_known_value(store, "vix_raw", vix_val)

    vix_score = None if vix_val is None else round(vix_to_score(vix_val), 1)

    # Put/Call (live -> last-known)
    pcr_val = get_put_call_ratio_equity()
    used_last_pcr = False
    if pcr_val is None:
        pcr_val, used_last_pcr = get_last_known_value(store, "putcall_raw", max_age_hours=96)
    if pcr_val is not None and not used_last_pcr:
        set_last_known_value(store, "putcall_raw", pcr_val)

    pcr_score = None if pcr_val is None else round(pcr_to_score(pcr_val), 1)

    # Trends (live -> last-known)
    trends_score = get_google_trends_score()
    used_last_trends = False
    if trends_score is None:
        trends_score, used_last_trends = get_last_known_value(store, "trends_score", max_age_hours=96)
    if trends_score is not None and not used_last_trends:
        set_last_known_value(store, "trends_score", trends_score)

    # Save last-known values
    save_last_known(store)

    # Composite
    composite, weights_used, comps_used, composite_note = compute_composite_score(
        cnn_score=cnn_score,
        reddit_score=reddit_score,
        vix_score=vix_score,
        pcr_score=pcr_score,
        trends_score=trends_score,
        reddit_weight_pct=reddit_weight
    )

# Coverage / Confidence
total_signals = 5
available_signals = sum([
    1 if cnn_score is not None else 0,
    1 if reddit_score is not None else 0,
    1 if vix_score is not None else 0,
    1 if pcr_score is not None else 0,
    1 if trends_score is not None else 0,
])
conf = confidence_label(available_signals, total_signals)

# ----------------------------
# Headline
# ----------------------------
st.markdown("## Today’s Market Mood")

hl, hr = st.columns([2, 1])
with hl:
    if composite is None:
        st.subheader("Not enough signals to produce a composite score today")
        st.write("Some sources are temporarily unavailable. Try again later.")
        if composite_note:
            st.caption(composite_note)
    else:
        st.metric("Composite score", f"{composite:.1f}")
        st.write(f"**Mood:** {market_label(composite)}")
        st.caption(f"Coverage: {available_signals}/{total_signals} signals • Confidence: {conf}")

with hr:
    st.markdown("**Signals used today**")
    used = []
    if cnn_score is not None: used.append("CNN")
    if reddit_score is not None: used.append("Reddit")
    if vix_score is not None: used.append("VIX")
    if pcr_score is not None: used.append("Put/Call")
    if trends_score is not None: used.append("Google Trends")
    st.write(", ".join(used) if used else "None")

# ----------------------------
# Key signals
# ----------------------------
st.markdown("## Key Signals")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("CNN Fear & Greed", "N/A" if cnn_score is None else f"{cnn_score:.1f}")
    st.caption(f"Label: {cnn_rating if cnn_rating else 'N/A'} • Source: CNN")

with c2:
    st.metric("Reddit mood", "N/A" if reddit_score is None else f"{reddit_score:.1f}")
    st.caption("Source: Reddit (overall market discussions)")

with c3:
    vix_label = "N/A" if vix_val is None else f"{float(vix_val):.2f}"
    st.metric("VIX (market calmness)", vix_label)
    if used_last_vix:
        st.caption("Showing last known value (up to 3 days)")
    else:
        st.caption("Lower VIX usually means calmer markets")

s1, s2 = st.columns(2)
with s1:
    pcr_label = "N/A" if pcr_val is None else f"{float(pcr_val):.3f}"
    st.metric("Options hedging (Put/Call)", pcr_label)
    st.caption("Higher Put/Call can mean more hedging / fear" + (" • Last known" if used_last_pcr else ""))

with s2:
    t_label = "N/A" if trends_score is None else f"{float(trends_score):.1f}"
    st.metric("Search interest (Google Trends)", t_label)
    st.caption("Compares fear vs greed search terms (last 7 days)" + (" • Last known" if used_last_trends else ""))

with st.expander("What went into today’s score"):
    rows = []
    def add_row(name, val):
        rows.append({
            "Signal": name,
            "Score (0–100)": "N/A" if val is None else round(float(val), 1),
            "Status": "Available" if val is not None else "Unavailable"
        })
    add_row("CNN", cnn_score)
    add_row("Reddit mood", reddit_score)
    add_row("VIX calmness", vix_score)
    add_row("Options hedging", pcr_score)
    add_row("Search interest", trends_score)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if composite is not None and weights_used:
        st.markdown("**How the composite is weighted today (auto-adjusts when something is missing)**")
        wdf = pd.DataFrame([{
            "Signal": k,
            "Weight": round(v, 3),
            "Signal score": round(comps_used.get(k, float("nan")), 1),
        } for k, v in weights_used.items()]).sort_values("Weight", ascending=False)
        st.dataframe(wdf, use_container_width=True)

# Reddit sections only if available
if reddit_score is not None:
    st.markdown("## Reddit Breakdown")
    sub_df = pd.DataFrame(subreddit_rows)
    st.dataframe(sub_df, use_container_width=True)

    chart_df = sub_df.dropna(subset=["score_0_100"])
    if not chart_df.empty:
        fig = px.bar(chart_df, x="subreddit", y="score_0_100", title="Reddit mood score by subreddit (0–100)")
        st.plotly_chart(fig, use_container_width=True)

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
            st.markdown("**Word cloud**")
            wc_text = " ".join(clean_text(t) for t in all_texts)
            wc = WordCloud(width=1200, height=450, background_color="white").generate(wc_text)
            fig_wc = plt.figure(figsize=(12, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig_wc, clear_figure=True)
else:
    st.info("Reddit mood is temporarily unavailable, so the Reddit breakdown is hidden for now.")

# Historical
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
    "updated_at": utc_now_iso(),
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
                title="Daily sentiment signals (0–100) and composite"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Run the analysis on different days to build a useful trend line.")
    except Exception:
        st.warning("Could not read the historical file.")

with st.expander("Details (why something might show N/A)"):
    st.write(
        "Sometimes a public source temporarily limits access. When that happens, we hide the signal and "
        "automatically adjust the composite to use only what’s available."
    )
    st.write("When possible, we show a recent last known value to keep the snapshot useful.")

col_r, col_c = st.columns([1, 1])
with col_r:
    if st.button("🔄 Clear cache and refresh"):
        st.cache_data.clear()
        st.rerun()

with col_c:
    if st.button("🧹 Clear history (danger)"):
        try:
            if os.path.exists(HISTORICAL_FILE):
                os.remove(HISTORICAL_FILE)
            st.success("History cleared.")
            st.rerun()
        except Exception:
            st.error("Could not clear history file.")

st.caption("Data sources: CNN, Reddit, CBOE, Google Trends, Yahoo Finance • Educational only")
