"""
╔══════════════════════════════════════════════════════════════╗
║        AI SENTIMENT ANALYSIS DASHBOARD — app.py             ║
║  Professional-grade NLP analytics platform with Streamlit   ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="SentimentIQ — AI Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject CSS Theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #1a2235;
    --border:    #2a3550;
    --pos:       #10d48e;
    --neg:       #ff4d6d;
    --neu:       #6b9fff;
    --amber:     #fbbf24;
    --text:      #e8edf8;
    --muted:     #8899bb;
    --font-head: 'DM Serif Display', serif;
    --font-body: 'Outfit', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Hero Header ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1e3d 0%, #1a0a2e 50%, #0a1a0f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(16,212,142,0.08), transparent),
                radial-gradient(ellipse 40% 60% at 20% 50%, rgba(107,159,255,0.06), transparent);
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-head) !important;
    font-size: 2.8rem;
    color: var(--text) !important;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(16,212,142,0.12);
    border: 1px solid rgba(16,212,142,0.3);
    color: var(--pos);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #3a4570; }
.metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 12px 12px 0 0;
}
.metric-card.total::after   { background: var(--neu); }
.metric-card.positive::after{ background: var(--pos); }
.metric-card.negative::after{ background: var(--neg); }
.metric-card.neutral::after { background: var(--amber); }
.metric-value {
    font-family: var(--font-head);
    font-size: 2.4rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label { color: var(--muted); font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-delta { font-size: 0.78rem; margin-top: 0.4rem; }

/* ── Result Box ── */
.result-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    position: relative;
}
.sentiment-label {
    font-family: var(--font-head);
    font-size: 2rem;
}
.score-bar-wrap { background: var(--border); border-radius: 100px; height: 6px; margin: 0.5rem 0 1rem; overflow: hidden; }
.score-bar { height: 100%; border-radius: 100px; transition: width 0.6s ease; }

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
thead tr th { background: var(--surface) !important; color: var(--muted) !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
tbody tr:nth-child(even) td { background: rgba(255,255,255,0.015) !important; }
tbody tr:hover td { background: rgba(107,159,255,0.06) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #10d48e, #0aad74) !important;
    color: #001a0e !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(16,212,142,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(16,212,142,0.35) !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--pos) !important;
    box-shadow: 0 0 0 2px rgba(16,212,142,0.15) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .sidebar-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--text) !important;
}

/* ── Misc ── */
hr { border-color: var(--border) !important; }
.section-head {
    font-family: var(--font-head);
    font-size: 1.4rem;
    margin: 1.5rem 0 1rem;
    color: var(--text);
}
.tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: var(--font-mono);
}
.tag-pos { background: rgba(16,212,142,0.15); color: var(--pos); border: 1px solid rgba(16,212,142,0.3); }
.tag-neg { background: rgba(255,77,109,0.15); color: var(--neg); border: 1px solid rgba(255,77,109,0.3); }
.tag-neu { background: rgba(107,159,255,0.15); color: var(--neu); border: 1px solid rgba(107,159,255,0.3); }

.stAlert { border-radius: 10px !important; }
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  IMPORTS — lazy-loaded with graceful fallbacks
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nlp_libs():
    """Load heavy NLP libraries once and cache."""
    import nltk
    for pkg in ["vader_lexicon", "stopwords", "punkt", "wordnet",
                "punkt_tab", "averaged_perceptron_tagger"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from textblob import TextBlob
    vader = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    return vader, lemmatizer, stop_words, TextBlob

vader, lemmatizer, stop_words, TextBlob = load_nlp_libs()


# ─────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # mentions/hashtags
    text = re.sub(r"[^a-z\s']", " ", text)              # punctuation
    text = re.sub(r"\s+", " ", text).strip()

    import nltk
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()

    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
#  SENTIMENT ENGINE
# ─────────────────────────────────────────────────────────────
def analyze_sentiment(text: str, engine: str = "VADER+TextBlob") -> dict:
    """
    Run sentiment analysis with selected engine.
    Returns label, polarity, confidence, scores dict.
    """
    clean = preprocess_text(text)
    if not clean.strip():
        clean = text  # fallback to raw if cleaning strips everything

    # --- VADER ---
    vs = vader.polarity_scores(text)
    vader_compound = vs["compound"]

    # --- TextBlob ---
    tb = TextBlob(text)
    tb_polarity = tb.sentiment.polarity        # -1 → +1
    tb_subjectivity = tb.sentiment.subjectivity

    # --- Ensemble (default) ---
    if engine == "VADER+TextBlob":
        compound = (vader_compound * 0.6) + (tb_polarity * 0.4)
        source = "VADER + TextBlob Ensemble"
    elif engine == "VADER":
        compound = vader_compound
        source = "VADER"
    else:  # TextBlob
        compound = tb_polarity
        source = "TextBlob"

    # Classify
    if compound >= 0.05:
        label, color, emoji = "POSITIVE", "#10d48e", "✅"
    elif compound <= -0.05:
        label, color, emoji = "NEGATIVE", "#ff4d6d", "❌"
    else:
        label, color, emoji = "NEUTRAL", "#6b9fff", "➖"

    # Confidence: distance from neutral threshold, scaled 0-100
    confidence = min(100, round(abs(compound) * 100 + 50))
    if confidence > 99:
        confidence = 99  # never claim 100%

    return {
        "label": label,
        "emoji": emoji,
        "color": color,
        "polarity": round(compound, 4),
        "confidence": confidence,
        "subjectivity": round(tb_subjectivity, 4),
        "vader_pos": round(vs["pos"], 3),
        "vader_neg": round(vs["neg"], 3),
        "vader_neu": round(vs["neu"], 3),
        "vader_compound": round(vader_compound, 4),
        "tb_polarity": round(tb_polarity, 4),
        "clean_text": clean,
        "source": source,
    }


def batch_analyze(texts: list, engine: str = "VADER+TextBlob") -> pd.DataFrame:
    """Analyze a list of texts and return a DataFrame."""
    rows = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            continue
        r = analyze_sentiment(t, engine)
        rows.append({
            "Index": i + 1,
            "Text": t[:120] + ("…" if len(t) > 120 else ""),
            "Sentiment": r["label"],
            "Polarity": r["polarity"],
            "Confidence %": r["confidence"],
            "Subjectivity": r["subjectivity"],
            "VADER Compound": r["vader_compound"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  SAMPLE DATASETS
# ─────────────────────────────────────────────────────────────
SAMPLE_DATASETS = {
    "Product Reviews": [
        "This product is absolutely amazing! Best purchase I've made all year.",
        "Terrible quality. Broke after two days. Complete waste of money.",
        "It's okay, nothing special. Does the job but could be better.",
        "Exceeded all my expectations! Fast delivery and perfect packaging.",
        "Disappointed with this item. The description was very misleading.",
        "Works fine. Average product for the price range.",
        "Phenomenal! Highly recommend to everyone looking for quality.",
        "Worst purchase ever. Customer service was also unhelpful.",
        "Decent product but nothing extraordinary. Would consider again.",
        "Outstanding quality! Will definitely buy from this seller again.",
        "Not worth the price at all. Very cheap build quality.",
        "Exactly as described. Happy with my purchase overall.",
    ],
    "Twitter / Social": [
        "Just watched the new movie - absolutely breathtaking! 🎬 #MustWatch",
        "Can't believe how bad the service was today. Never going back! 😡",
        "Today was just another Monday. Nothing exciting happened.",
        "The concert last night was INCREDIBLE! Best night of my life! 🎶",
        "Traffic is terrible as usual. Why do I even bother taking this route?",
        "Meh, the new update didn't really change anything important.",
        "So grateful for all the support from this amazing community! ❤️",
        "This app keeps crashing. So frustrating and such poor quality.",
        "Had lunch. It was fine. Going back to work now.",
        "Just got promoted! Dreams really do come true if you work hard! 🚀",
    ],
    "Customer Feedback": [
        "The support team resolved my issue within minutes. Impressive!",
        "I've been waiting 3 weeks for a refund. This is unacceptable.",
        "The onboarding process was straightforward and well documented.",
        "Your platform constantly goes down. It's costing me business.",
        "The new dashboard update is intuitive and looks great!",
        "Billing errors keep happening. Third time this month.",
        "Documentation is comprehensive and the API works as expected.",
        "Response time from support is way too slow for a paid service.",
        "Neutral about the changes. Some things improved, some regressed.",
        "Best SaaS tool I've used. The team really listens to feedback!",
    ],
}


# ─────────────────────────────────────────────────────────────
#  VISUALIZATION HELPERS (Plotly)
# ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px

PALETTE = {"POSITIVE": "#10d48e", "NEGATIVE": "#ff4d6d", "NEUTRAL": "#6b9fff"}
BG      = "#111827"
CARD_BG = "#1a2235"
TEXT_C  = "#e8edf8"
MUTED   = "#8899bb"
GRID    = "#2a3550"

BASE_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Outfit, sans-serif", color=TEXT_C),
    margin=dict(l=20, r=20, t=40, b=20),
)

def pie_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = go.Figure(go.Pie(
        labels=counts["Sentiment"],
        values=counts["Count"],
        marker_colors=[PALETTE[s] for s in counts["Sentiment"]],
        hole=0.52,
        textinfo="label+percent",
        textfont_size=13,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    total = counts["Count"].sum()
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:11px'>texts</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=22, color=TEXT_C))
    fig.update_layout(**BASE_LAYOUT, title=dict(text="Sentiment Distribution", font_size=15),
                      showlegend=True,
                      legend=dict(bgcolor=CARD_BG, bordercolor=GRID, borderwidth=1))
    return fig


def bar_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Sentiment"].value_counts().reindex(["POSITIVE", "NEGATIVE", "NEUTRAL"], fill_value=0)
    fig = go.Figure()
    for label, val in counts.items():
        fig.add_trace(go.Bar(
            x=[label], y=[val],
            name=label,
            marker_color=PALETTE[label],
            marker_line_width=0,
            width=0.55,
            text=[val], textposition="outside",
            textfont=dict(size=14, color=TEXT_C),
            hovertemplate=f"<b>{label}</b><br>Count: {val}<extra></extra>",
        ))
    fig.update_layout(**BASE_LAYOUT,
                      title=dict(text="Sentiment Count", font_size=15),
                      xaxis=dict(showgrid=False, color=MUTED),
                      yaxis=dict(gridcolor=GRID, color=MUTED, title="Count"),
                      showlegend=False, barmode="group")
    return fig


def polarity_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in PALETTE.items():
        sub = df[df["Sentiment"] == label]["Polarity"]
        if len(sub):
            fig.add_trace(go.Histogram(
                x=sub, name=label, marker_color=color,
                opacity=0.8, nbinsx=20,
                hovertemplate=f"<b>{label}</b><br>Polarity: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
            ))
    fig.update_layout(**BASE_LAYOUT,
                      title="Polarity Score Distribution",
                      barmode="overlay",
                      xaxis=dict(title="Polarity Score", gridcolor=GRID, color=MUTED, range=[-1.1, 1.1]),
                      yaxis=dict(title="Frequency", gridcolor=GRID, color=MUTED),
                      legend=dict(bgcolor=CARD_BG, bordercolor=GRID))
    return fig


def confidence_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in PALETTE.items():
        sub = df[df["Sentiment"] == label]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["Polarity"], y=sub["Confidence %"],
                mode="markers", name=label,
                marker=dict(color=color, size=9, opacity=0.75,
                            line=dict(width=1, color="rgba(255,255,255,0.15)")),
                hovertemplate="<b>%{text}</b><br>Polarity: %{x:.3f}<br>Confidence: %{y}%<extra></extra>",
                text=sub["Text"].str[:50],
            ))
    fig.update_layout(**BASE_LAYOUT,
                      title="Polarity vs Confidence",
                      xaxis=dict(title="Polarity", gridcolor=GRID, color=MUTED, zeroline=True,
                                 zerolinecolor=GRID, range=[-1.1, 1.1]),
                      yaxis=dict(title="Confidence %", gridcolor=GRID, color=MUTED),
                      legend=dict(bgcolor=CARD_BG, bordercolor=GRID))
    return fig


def trend_chart(df: pd.DataFrame) -> go.Figure:
    """Simulated time-series trend (uses index as proxy time)."""
    df2 = df.copy().reset_index(drop=True)
    df2["Time"] = [datetime.now() - timedelta(hours=len(df2)-i) for i in range(len(df2))]
    df2["Score"] = df2["Polarity"].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()
    # Fill area
    fig.add_trace(go.Scatter(
        x=df2["Time"], y=df2["Score"],
        fill="tozeroy",
        fillcolor="rgba(16,212,142,0.08)",
        line=dict(color="#10d48e", width=2),
        name="Avg Polarity",
        hovertemplate="Time: %{x|%H:%M}<br>Score: %{y:.3f}<extra></extra>",
    ))
    # Scatter points colored by sentiment
    for label, color in PALETTE.items():
        sub = df2[df2["Sentiment"] == label]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["Time"], y=sub["Polarity"],
                mode="markers", name=label,
                marker=dict(color=color, size=7, opacity=0.85),
            ))
    fig.update_layout(**BASE_LAYOUT,
                      title="Sentiment Trend Over Time",
                      xaxis=dict(gridcolor=GRID, color=MUTED),
                      yaxis=dict(gridcolor=GRID, color=MUTED, title="Polarity Score",
                                 zeroline=True, zerolinecolor=GRID),
                      legend=dict(bgcolor=CARD_BG, bordercolor=GRID))
    return fig


def wordcloud_figure(texts: list, sentiment_filter: str = "ALL") -> "matplotlib.figure.Figure":
    """Generate a styled word cloud."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    combined = " ".join([preprocess_text(t) for t in texts if isinstance(t, str)])
    if not combined.strip():
        combined = "no text available"

    color_map = {
        "POSITIVE": ["#10d48e", "#0aad74", "#05ffb0", "#b0ffe0"],
        "NEGATIVE": ["#ff4d6d", "#ff1a40", "#ff8098", "#ffb3bf"],
        "NEUTRAL":  ["#6b9fff", "#3d7eff", "#a0bfff", "#c8daff"],
        "ALL":      ["#10d48e", "#6b9fff", "#fbbf24", "#ff4d6d", "#a78bfa"],
    }
    colors = color_map.get(sentiment_filter, color_map["ALL"])

    def color_func(*args, **kwargs):
        import random
        return random.choice(colors)

    wc = WordCloud(
        width=900, height=420,
        background_color="#1a2235",
        max_words=100,
        color_func=color_func,
        prefer_horizontal=0.8,
        collocations=False,
        min_font_size=11,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#1a2235")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of single-text results
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
      <div style='font-family:"DM Serif Display",serif; font-size:1.5rem; color:#e8edf8;'>SentimentIQ</div>
      <div style='color:#8899bb; font-size:0.78rem; letter-spacing:0.08em;'>AI ANALYTICS ENGINE</div>
    </div>
    <hr style='border-color:#2a3550; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    # Engine selector
    st.markdown("**⚙️ Analysis Engine**")
    engine = st.selectbox(
        "Model", ["VADER+TextBlob", "VADER", "TextBlob"],
        help="VADER excels at social media; TextBlob at formal text. Ensemble combines both."
    )

    st.markdown("---")

    # Stats from session
    if st.session_state.history:
        labels = [h["label"] for h in st.session_state.history]
        pos = labels.count("POSITIVE")
        neg = labels.count("NEGATIVE")
        neu = labels.count("NEUTRAL")
        st.markdown("**📊 Session Stats**")
        st.markdown(f"""
        <div style='display:flex; flex-direction:column; gap:0.4rem;'>
          <div style='display:flex; justify-content:space-between;'>
            <span style='color:#8899bb;'>Total Analyzed</span>
            <span style='color:#e8edf8; font-weight:600;'>{len(labels)}</span>
          </div>
          <div style='display:flex; justify-content:space-between;'>
            <span style='color:#10d48e;'>● Positive</span>
            <span style='color:#10d48e; font-weight:600;'>{pos}</span>
          </div>
          <div style='display:flex; justify-content:space-between;'>
            <span style='color:#ff4d6d;'>● Negative</span>
            <span style='color:#ff4d6d; font-weight:600;'>{neg}</span>
          </div>
          <div style='display:flex; justify-content:space-between;'>
            <span style='color:#6b9fff;'>● Neutral</span>
            <span style='color:#6b9fff; font-weight:600;'>{neu}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.session_state.batch_df = None
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:#8899bb; font-size:0.75rem; line-height:1.6;'>
    <b style='color:#e8edf8;'>About</b><br>
    Powered by VADER, TextBlob, NLTK & Streamlit.<br><br>
    Supports manual input, CSV upload, and built-in demo datasets.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🧠 NLP Intelligence Platform</div>
  <div class="hero-title">SentimentIQ Dashboard</div>
  <div class="hero-subtitle">Real-time AI-powered sentiment analysis for reviews, social media, and customer feedback</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  GLOBAL METRICS (from batch_df or history)
# ─────────────────────────────────────────────────────────────
def render_metrics(df_or_list):
    if isinstance(df_or_list, pd.DataFrame) and len(df_or_list):
        total = len(df_or_list)
        pos   = (df_or_list["Sentiment"] == "POSITIVE").sum()
        neg   = (df_or_list["Sentiment"] == "NEGATIVE").sum()
        neu   = (df_or_list["Sentiment"] == "NEUTRAL").sum()
        avg_pol = df_or_list["Polarity"].mean()
    elif isinstance(df_or_list, list) and df_or_list:
        total   = len(df_or_list)
        pos     = sum(1 for h in df_or_list if h["label"] == "POSITIVE")
        neg     = sum(1 for h in df_or_list if h["label"] == "NEGATIVE")
        neu     = sum(1 for h in df_or_list if h["label"] == "NEUTRAL")
        avg_pol = np.mean([h["polarity"] for h in df_or_list])
    else:
        total = pos = neg = neu = 0
        avg_pol = 0.0

    pct_pos = round(pos / total * 100) if total else 0
    pct_neg = round(neg / total * 100) if total else 0

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card total">
        <div class="metric-value" style="color:#6b9fff;">{total}</div>
        <div class="metric-label">Total Analyzed</div>
        <div class="metric-delta" style="color:#8899bb;">Avg polarity: {avg_pol:.3f}</div>
      </div>
      <div class="metric-card positive">
        <div class="metric-value" style="color:#10d48e;">{pos}</div>
        <div class="metric-label">Positive</div>
        <div class="metric-delta" style="color:#10d48e;">{pct_pos}% of total</div>
      </div>
      <div class="metric-card negative">
        <div class="metric-value" style="color:#ff4d6d;">{neg}</div>
        <div class="metric-label">Negative</div>
        <div class="metric-delta" style="color:#ff4d6d;">{pct_neg}% of total</div>
      </div>
      <div class="metric-card neutral">
        <div class="metric-value" style="color:#fbbf24;">{neu}</div>
        <div class="metric-label">Neutral</div>
        <div class="metric-delta" style="color:#fbbf24;">{total - pos - neg}% mixed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# Show metrics from current data source
active_data = st.session_state.batch_df if st.session_state.batch_df is not None else st.session_state.history
render_metrics(active_data)


# ─────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Single Analysis",
    "📦  Batch Analysis",
    "📊  Visualizations",
    "📋  History & Export",
])


# ══════════════════════════════════════════════
#  TAB 1 — SINGLE TEXT ANALYSIS
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">Analyze a Single Text</div>', unsafe_allow_html=True)

    col_in, col_ex = st.columns([3, 1])
    with col_in:
        text_input = st.text_area(
            "Enter text to analyze",
            placeholder="Type or paste any text — review, tweet, comment, sentence…",
            height=130,
            label_visibility="collapsed",
        )
    with col_ex:
        st.markdown("**Quick examples:**")
        examples = {
            "😊 Positive": "This product is absolutely amazing! Exceeded all my expectations.",
            "😞 Negative": "Terrible experience. The worst customer service I've ever had.",
            "😐 Neutral":  "The package arrived today. It contains the items I ordered.",
            "🤔 Mixed":    "The food was great but the service was unbelievably slow.",
        }
        for label, ex in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state["quick_example"] = ex
                st.rerun()

    # Apply quick example
    if "quick_example" in st.session_state:
        text_input = st.session_state.pop("quick_example")

    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=False)

    if analyze_btn and text_input.strip():
        with st.spinner("Processing…"):
            time.sleep(0.3)  # UX pause
            result = analyze_sentiment(text_input, engine)
            st.session_state.history.append({**result, "text": text_input, "ts": datetime.now()})

        c_color = result["color"]
        bar_pct = round((result["polarity"] + 1) / 2 * 100)

        st.markdown(f"""
        <div class="result-box">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
            <div>
              <div style="color:{c_color}; font-size:0.8rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem;">
                {result['emoji']} Detected Sentiment
              </div>
              <div class="sentiment-label" style="color:{c_color};">{result['label']}</div>
              <div style="margin-top:0.5rem; color:#8899bb; font-size:0.85rem;">via {result['source']}</div>
            </div>
            <div style="display:flex; gap:2.5rem; flex-wrap:wrap;">
              <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:700; color:{c_color}; font-family:'DM Mono',monospace;">{result['confidence']}%</div>
                <div style="color:#8899bb; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Confidence</div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:700; color:#e8edf8; font-family:'DM Mono',monospace;">{result['polarity']:.3f}</div>
                <div style="color:#8899bb; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Polarity</div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:700; color:#fbbf24; font-family:'DM Mono',monospace;">{result['subjectivity']:.2f}</div>
                <div style="color:#8899bb; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Subjectivity</div>
              </div>
            </div>
          </div>

          <div style="margin-top:1.2rem;">
            <div style="display:flex; justify-content:space-between; color:#8899bb; font-size:0.75rem; margin-bottom:4px;">
              <span>← Very Negative</span><span>Very Positive →</span>
            </div>
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{bar_pct}%; background:{c_color};"></div>
            </div>
          </div>

          <div style="display:flex; gap:1rem; margin-top:1rem; flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.6rem 1rem; flex:1; min-width:120px;">
              <div style="color:#8899bb; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">VADER +</div>
              <div style="color:#10d48e; font-family:'DM Mono',monospace; font-size:0.9rem;">{result['vader_pos']}</div>
            </div>
            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.6rem 1rem; flex:1; min-width:120px;">
              <div style="color:#8899bb; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">VADER −</div>
              <div style="color:#ff4d6d; font-family:'DM Mono',monospace; font-size:0.9rem;">{result['vader_neg']}</div>
            </div>
            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.6rem 1rem; flex:1; min-width:120px;">
              <div style="color:#8899bb; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">VADER =</div>
              <div style="color:#6b9fff; font-family:'DM Mono',monospace; font-size:0.9rem;">{result['vader_neu']}</div>
            </div>
            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:0.6rem 1rem; flex:1; min-width:140px;">
              <div style="color:#8899bb; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;">TextBlob</div>
              <div style="color:#fbbf24; font-family:'DM Mono',monospace; font-size:0.9rem;">{result['tb_polarity']}</div>
            </div>
          </div>

          <div style="margin-top:1rem; padding:0.8rem 1rem; background:rgba(255,255,255,0.03); border-radius:8px;">
            <span style="color:#8899bb; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Cleaned Text: </span>
            <span style="color:#8899bb; font-family:'DM Mono',monospace; font-size:0.82rem;">{result['clean_text'][:200]}{'…' if len(result['clean_text'])>200 else ''}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please enter some text before analyzing.")


# ══════════════════════════════════════════════
#  TAB 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Batch Analysis</div>', unsafe_allow_html=True)

    col_ds, col_up = st.columns([1, 1])

    with col_ds:
        st.markdown("**📚 Use a Demo Dataset**")
        dataset_choice = st.selectbox("Choose dataset", list(SAMPLE_DATASETS.keys()))
        if st.button("▶ Analyze Dataset", use_container_width=True):
            texts = SAMPLE_DATASETS[dataset_choice]
            with st.spinner(f"Analyzing {len(texts)} texts…"):
                st.session_state.batch_df = batch_analyze(texts, engine)
                # Also add to history
                for t in texts:
                    r = analyze_sentiment(t, engine)
                    st.session_state.history.append({**r, "text": t, "ts": datetime.now()})
            st.success(f"✅ Analyzed {len(texts)} texts!")
            st.rerun()

    with col_up:
        st.markdown("**📂 Upload CSV File**")
        uploaded = st.file_uploader(
            "CSV with a 'text' column (or first column used)",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                text_col = "text" if "text" in df_upload.columns else df_upload.columns[0]
                texts_from_csv = df_upload[text_col].dropna().tolist()
                st.info(f"Found **{len(texts_from_csv)}** rows in column `{text_col}`.")
                if st.button("▶ Analyze CSV", use_container_width=True):
                    with st.spinner(f"Analyzing {len(texts_from_csv)} texts…"):
                        st.session_state.batch_df = batch_analyze(texts_from_csv, engine)
                    st.success("Done!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Also allow multi-line text box
    st.markdown("---")
    st.markdown("**✏️ Paste Multiple Texts** (one per line)")
    multi_text = st.text_area("One text per line", height=120, label_visibility="collapsed",
                              placeholder="Paste texts here, one per line…")
    if st.button("▶ Analyze All Lines"):
        lines = [l.strip() for l in multi_text.splitlines() if l.strip()]
        if lines:
            with st.spinner(f"Analyzing {len(lines)} lines…"):
                st.session_state.batch_df = batch_analyze(lines, engine)
            st.success(f"✅ Analyzed {len(lines)} texts!")
            st.rerun()
        else:
            st.warning("No text found.")

    # Display results table
    if st.session_state.batch_df is not None and len(st.session_state.batch_df):
        df_show = st.session_state.batch_df.copy()
        st.markdown("---")
        st.markdown(f'<div class="section-head">Results — {len(df_show)} texts</div>', unsafe_allow_html=True)

        # Filter controls
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            sent_filter = st.multiselect("Filter sentiment", ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                                         default=["POSITIVE", "NEGATIVE", "NEUTRAL"])
        with fc2:
            sort_by = st.selectbox("Sort by", ["Index", "Polarity", "Confidence %"])
        with fc3:
            search_q = st.text_input("Search text", placeholder="Filter by keyword…")

        filtered = df_show[df_show["Sentiment"].isin(sent_filter)]
        if search_q:
            filtered = filtered[filtered["Text"].str.contains(search_q, case=False, na=False)]
        filtered = filtered.sort_values(sort_by, ascending=(sort_by == "Index"))

        # Color sentiment column
        def color_sentiment(val):
            colors = {"POSITIVE": "color: #10d48e", "NEGATIVE": "color: #ff4d6d", "NEUTRAL": "color: #6b9fff"}
            return colors.get(val, "")

        styled = filtered.style.applymap(color_sentiment, subset=["Sentiment"])
        st.dataframe(styled, use_container_width=True, height=380)


# ══════════════════════════════════════════════
#  TAB 3 — VISUALIZATIONS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">Visual Analytics</div>', unsafe_allow_html=True)

    # Determine data source
    viz_df = st.session_state.batch_df
    if viz_df is None and st.session_state.history:
        # Build df from history
        viz_df = pd.DataFrame([{
            "Text": h["text"][:100],
            "Sentiment": h["label"],
            "Polarity": h["polarity"],
            "Confidence %": h["confidence"],
            "Subjectivity": h["subjectivity"],
            "VADER Compound": h["vader_compound"],
        } for h in st.session_state.history])

    if viz_df is None or len(viz_df) == 0:
        st.info("💡 Analyze some texts first (in Single or Batch tabs) to see visualizations.")
        # Load default dataset for demo
        if st.button("🎲 Load Demo Data for Visualization"):
            demo_texts = SAMPLE_DATASETS["Product Reviews"] + SAMPLE_DATASETS["Twitter / Social"]
            st.session_state.batch_df = batch_analyze(demo_texts, engine)
            st.rerun()
    else:
        # Row 1: Pie + Bar
        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(pie_chart(viz_df), use_container_width=True)
        with vc2:
            st.plotly_chart(bar_chart(viz_df), use_container_width=True)

        # Row 2: Histogram + Scatter
        vc3, vc4 = st.columns(2)
        with vc3:
            st.plotly_chart(polarity_histogram(viz_df), use_container_width=True)
        with vc4:
            st.plotly_chart(confidence_scatter(viz_df), use_container_width=True)

        # Row 3: Trend (full width)
        if len(viz_df) >= 3:
            st.plotly_chart(trend_chart(viz_df), use_container_width=True)

        # Row 4: Word Cloud
        st.markdown("---")
        wc_col1, wc_col2 = st.columns([3, 1])
        with wc_col2:
            wc_filter = st.selectbox("Word Cloud — filter", ["ALL", "POSITIVE", "NEGATIVE", "NEUTRAL"])
        with wc_col1:
            st.markdown(f'<div class="section-head">Word Cloud ({wc_filter})</div>', unsafe_allow_html=True)

        try:
            if wc_filter == "ALL":
                wc_texts = viz_df["Text"].tolist()
            else:
                wc_texts = viz_df[viz_df["Sentiment"] == wc_filter]["Text"].tolist()

            if wc_texts:
                import matplotlib
                matplotlib.use("Agg")
                fig_wc = wordcloud_figure(wc_texts, wc_filter)
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.info(f"No {wc_filter} texts found for word cloud.")
        except ImportError:
            st.warning("wordcloud package not installed. Run: `pip install wordcloud`")
        except Exception as e:
            st.error(f"Word cloud error: {e}")


# ══════════════════════════════════════════════
#  TAB 4 — HISTORY & EXPORT
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-head">Analysis History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analysis history yet. Start analyzing texts in the other tabs!")
    else:
        hist_df = pd.DataFrame([{
            "Timestamp": h["ts"].strftime("%H:%M:%S"),
            "Text": h["text"][:90] + ("…" if len(h["text"]) > 90 else ""),
            "Sentiment": h["label"],
            "Polarity": h["polarity"],
            "Confidence %": h["confidence"],
        } for h in reversed(st.session_state.history)])

        def color_sentiment(val):
            colors = {"POSITIVE": "color: #10d48e", "NEGATIVE": "color: #ff4d6d", "NEUTRAL": "color: #6b9fff"}
            return colors.get(val, "")

        st.dataframe(
            hist_df.style.applymap(color_sentiment, subset=["Sentiment"]),
            use_container_width=True,
            height=420,
        )

        st.markdown("---")
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Export History CSV",
                data=csv_data,
                file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with ec2:
            if st.session_state.batch_df is not None:
                csv_batch = st.session_state.batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Export Batch Results CSV",
                    data=csv_batch,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with ec3:
            # Summary stats
            total  = len(st.session_state.history)
            pos    = sum(1 for h in st.session_state.history if h["label"] == "POSITIVE")
            neg    = sum(1 for h in st.session_state.history if h["label"] == "NEGATIVE")
            neu    = total - pos - neg
            avg_p  = np.mean([h["polarity"] for h in st.session_state.history])
            avg_c  = np.mean([h["confidence"] for h in st.session_state.history])

            summary_txt = (
                f"SentimentIQ — Analysis Summary\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{'─'*40}\n"
                f"Total Texts Analyzed : {total}\n"
                f"Positive             : {pos} ({round(pos/total*100)}%)\n"
                f"Negative             : {neg} ({round(neg/total*100)}%)\n"
                f"Neutral              : {neu} ({round(neu/total*100)}%)\n"
                f"Average Polarity     : {avg_p:.4f}\n"
                f"Average Confidence   : {avg_c:.1f}%\n"
                f"Engine Used          : {engine}\n"
            )
            st.download_button(
                "⬇ Export Summary TXT",
                data=summary_txt.encode(),
                file_name="sentiment_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Summary insight box
        if total >= 3:
            dom = "POSITIVE" if pos > neg and pos > neu else ("NEGATIVE" if neg > pos and neg > neu else "NEUTRAL")
            dom_color = PALETTE[dom]
            st.markdown(f"""
            <div style='margin-top:1.5rem; background:rgba(255,255,255,0.03); border:1px solid #2a3550; border-radius:12px; padding:1.2rem 1.5rem;'>
              <div style='color:#8899bb; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>🤖 AI Insight Summary</div>
              <div style='color:#e8edf8; line-height:1.7; font-size:0.95rem;'>
                Across <b>{total}</b> analyzed texts, the dominant tone is
                <b style='color:{dom_color};'>{dom}</b>
                ({round(max(pos,neg,neu)/total*100)}% of texts).
                Average polarity of <b>{avg_p:.3f}</b> indicates a
                {'strongly positive' if avg_p > 0.3 else ('moderately positive' if avg_p > 0.05 else ('strongly negative' if avg_p < -0.3 else ('moderately negative' if avg_p < -0.05 else 'largely neutral')))}
                corpus. Average confidence of <b>{avg_c:.1f}%</b> suggests
                {'high model certainty' if avg_c > 75 else 'moderate model certainty'} in these predictions.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2.5rem 0 1rem; color:#3a4570; font-size:0.8rem;'>
  SentimentIQ · Built with Streamlit · VADER + TextBlob + NLTK<br>
  <span style='color:#2a3550;'>Professional AI Analytics Dashboard</span>
</div>
""", unsafe_allow_html=True)
