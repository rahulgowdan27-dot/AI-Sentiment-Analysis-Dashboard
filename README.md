# 🧠 SentimentIQ — AI Sentiment Analysis Dashboard

A professional-grade, full-stack NLP analytics platform built with Python and Streamlit.

---

## 🚀 Quick Start

### Option A — Run Locally

```bash
# 1. Clone / unzip the project
cd sentiment_dashboard/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

### Option B — Google Colab

1. Open `SentimentIQ_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `app.py` to Colab files
3. Run all cells top-to-bottom
4. Click the **localtunnel URL** that appears in the final cell

---

## 📁 Project Structure

```
sentiment_dashboard/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── SentimentIQ_Colab.ipynb   ← Google Colab notebook
└── README.md                 ← This file
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Single Analysis** | Analyze any text in real-time with detailed scores |
| **Batch Analysis** | Upload CSV or use built-in demo datasets |
| **Multi-Engine** | VADER, TextBlob, or ensemble mode |
| **Preprocessing** | Lowercase, stopword removal, lemmatization |
| **Visualizations** | Pie chart, bar chart, polarity histogram, scatter, trend, word cloud |
| **Export** | Download results as CSV + summary TXT |
| **History** | Full session history with AI insight summary |

---

## 🔬 Sentiment Engines

| Engine | Best For | Notes |
|---|---|---|
| **VADER** | Social media, reviews | Rule-based, handles emojis & slang |
| **TextBlob** | Formal text, articles | Pattern-based, good subjectivity scores |
| **VADER+TextBlob** | General purpose | 60/40 weighted ensemble |

### Classification Thresholds

```
Polarity ≥  0.05  →  POSITIVE ✅
Polarity ≤ -0.05  →  NEGATIVE ❌
Otherwise          →  NEUTRAL  ➖
```

---

## 📊 Output Format

```
Input    : "This product is amazing and works perfectly."
─────────────────────────────────────────────────────────
Sentiment  : ✅ POSITIVE
Confidence : 94%
Polarity   : +0.870
Subjectivity: 0.72
VADER +/-/= : 0.488 / 0.000 / 0.512
TextBlob   : +0.800
Cleaned    : product amazing work perfectly
```

---

## 📦 Dependencies

```
streamlit      ≥ 1.32   — Dashboard framework
nltk           ≥ 3.8    — Tokenization, stopwords, lemmatization
textblob       ≥ 0.17   — Sentiment analysis engine
vaderSentiment ≥ 3.3    — Social media sentiment
scikit-learn   ≥ 1.3    — ML utilities
plotly         ≥ 5.18   — Interactive charts
matplotlib     ≥ 3.7    — Word cloud rendering
wordcloud      ≥ 1.9    — Word frequency visualization
pandas         ≥ 2.0    — Data manipulation
numpy          ≥ 1.24   — Numerical operations
```

---

## 🎨 Dashboard Tabs

1. **🔍 Single Analysis** — Real-time text analysis with full score breakdown
2. **📦 Batch Analysis** — CSV upload or demo dataset analysis with filterable table
3. **📊 Visualizations** — 6 interactive charts + word cloud
4. **📋 History & Export** — Session history, AI summary, CSV/TXT export

---

## 💡 CSV Upload Format

Your CSV should have a column named `text` (or the first column is used):

```csv
text
"This product is amazing!"
"Terrible service, never again."
"It's an average experience."
```

---

## 🛠 Architecture

```
User Input / CSV
      ↓
Text Preprocessing (NLTK)
  • Lowercase → URL removal → Punctuation strip
  • Tokenization → Stopword filter → Lemmatization
      ↓
Sentiment Engine (VADER + TextBlob)
  • Compound polarity score
  • Confidence calculation
  • Label classification
      ↓
Dashboard Visualization (Streamlit + Plotly)
  • Real-time metrics
  • Interactive charts
  • Word clouds
  • Export options
```

---

*Built with Python · VADER · TextBlob · NLTK · Streamlit · Plotly*
