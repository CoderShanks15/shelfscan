<div align="center">

# 🛒 ShelfScan

### _A 3-second health verdict on any supermarket product._

Point your phone at a barcode. A 5-stage computer vision pipeline decodes it. A machine learning model scores the product's health from 0 to 100. CLIP analyses the product image. A price intelligence engine calculates how much nutrition you're getting per penny. All in under 3 seconds.

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Meta × MLH Hackathon 2025**

[Live Demo](#) · [Watch the Demo Video](#) · [Try These Barcodes](#-try-it-now)

</div>

---

## 🧠 Why I Built This

Every week, millions of people stand in supermarket aisles holding two similar products at different prices, unable to tell which is actually worth buying. They open Google. They get blog posts. They give up and guess.

I wanted to build something that gives a **direct, explainable answer in under 3 seconds** — not a search result, but a verdict. A number. A reason.

ShelfScan is that tool. Scan any product barcode with your phone. Get a health score from 0 to 100, an explanation of every point gained or lost, a CLIP-powered image classification, and a health-per-penny comparison — all from one photo.

The deeper motivation: **health information should not require nutritional expertise to decode.** Nutri-Score grades, NOVA groups, E-numbers — these exist but most people don't know what they mean. ShelfScan translates them into a number anyone can understand.

---

## ⚡ What It Does

| Feature | Description |
|---|---|
| 📸 **Barcode Scanning** | 5-stage CV pipeline decodes barcodes from real-world photos, including blurry, rotated, or dark images |
| 🧬 **Health Scoring** | GradientBoosting ML model scores products 0–100 using 9 nutrient features |
| 🔍 **Explainability** | Every score adjustment is logged — you see exactly why a product gained or lost points |
| 🖼️ **Image Classification** | CLIP zero-shot model classifies the product image without any training data |
| 💰 **Price Intelligence** | Calculates health-per-penny so you can compare value across products |
| ⚖️ **Compare Mode** | Side-by-side comparison of any two products |
| 👤 **User Accounts** | JWT auth, scan history, personalised scoring based on dietary profile |

---

## 🎯 Try It Now

These barcodes work instantly on the live demo — no camera needed:

| Product | Barcode | Score | Verdict |
|---|---|---|---|
| Nutella 400g | `3017620422003` | ~28 | ⚠️ Caution |
| Pringles Original | `5053990101538` | ~22 | ❌ Avoid |
| Kellogg's Corn Flakes | `5010477348549` | ~45 | 🟡 OK |
| Quaker Oats 1kg | `5000173008065` | ~74 | ✅ Great |
| Evian Water 1.5L | `3068320113994` | ~88 | ✅ Great |
| Haribo Gold-Bears | `4001686323564` | ~32 | ⚠️ Caution |

> Scan Nutella, then Evian Water, then hit Compare. The health-per-penny difference is a factor of 8.

---

## 🏗️ Architecture

### The Pipeline: Photo → Verdict

```
📱 User uploads photo or types barcode
              │
              ▼
┌─────────────────────────────────┐
│     COMPUTER VISION LAYER       │
│                                 │
│  OpenCV detects barcode region  │
│  (Scharr gradient + morphology) │
│              │                  │
│  Stage 1: pyzbar direct decode  │
│  Stage 2: 4-angle rotation scan │
│  Stage 3: CLAHE preprocessing   │
│  Stage 4: Adaptive threshold    │
│  Stage 5: zxing-cpp fallback    │
└──────────────┬──────────────────┘
               │  EAN-13 string
               ▼
┌─────────────────────────────────┐
│       DATA LAYER                │
│                                 │
│  Open Food Facts API            │
│  3M+ products, no API key       │
│  600+ fields per product        │
│  Cached via @st.cache_data      │
└──────────────┬──────────────────┘
               │  Product JSON
               ▼
┌─────────────────────────────────┐
│     MACHINE LEARNING LAYER      │
│                                 │
│  9 features extracted:          │
│  energy · fat · saturated fat   │
│  sugar · fiber · protein        │
│  salt · nova_group · additives  │
│              │                  │
│  GradientBoostingRegressor      │
│  StandardScaler + Pipeline      │
│              │                  │
│  Rule-based adjustments logged: │
│  organic +5 · ultra-proc −15    │
│  high additives −12 · etc.      │
└──────────────┬──────────────────┘
               │  Score + trail
               ▼
┌─────────────────────────────────┐
│      IMAGE ANALYSIS LAYER       │
│                                 │
│  CLIP (openai/clip-vit-base)    │
│  Zero-shot classification       │
│  No training data needed        │
│  Categories: fresh produce,     │
│  packaged, ultra-processed etc. │
└──────────────┬──────────────────┘
               │  Category + confidence
               ▼
┌─────────────────────────────────┐
│    PRICE INTELLIGENCE LAYER     │
│                                 │
│  Health-per-penny score         │
│  Nutrition density index        │
│  Category benchmark comparison  │
└──────────────┬──────────────────┘
               │
               ▼
        🖥️ Streamlit UI
   Tabs · Score bars · Compare mode
```

---

## 🔬 Technical Decisions & Why

### Why Gradient Boosting instead of a neural network?

Interpretability. Every feature's contribution to the score is logged and shown in the UI. When a product scores 28/100, the user sees: *"−12 for 8 additives, −8 for NOVA group 4, −5 for saturated fat, +3 for some fibre."* A neural network gives you a number. GBM gives you a reason. For a health tool, that difference matters more than marginal accuracy gains.

### How does CLIP classify images without training data?

CLIP (Contrastive Language–Image Pretraining) was trained by OpenAI on 400 million image-text pairs from the internet. It learns a shared embedding space where images and the text that describes them end up geometrically close together. ShelfScan gives it a product photo and a list of candidate labels — *"fresh produce"*, *"ultra-processed packaging"*, *"dairy product"* — and CLIP scores which label best matches the image. Zero training data from me. Zero labelling. Just prompts.

### Why a 5-stage barcode cascade?

Real-world photos are messy. Barcodes are blurry, angled, dark, or partially obscured. Each stage of the pipeline handles a different failure mode:

| Stage | Handles |
|---|---|
| pyzbar direct | Clean, well-lit barcodes |
| 4-angle rotation | Tilted or rotated barcodes |
| CLAHE preprocessing | Dark or low-contrast photos |
| Adaptive threshold | Uneven lighting across barcode |
| zxing-cpp fallback | Barcodes pyzbar structurally misses (~12% of real-world cases) |

### Why Streamlit and not Flask/FastAPI?

For a hackathon, Streamlit compresses weeks of frontend work into hours. `st.file_uploader` becomes a camera button on mobile with zero extra code. `@st.cache_resource` means the 340MB CLIP model loads once per server session and is reused across all users. The tradeoff is concurrency — for production at scale, a FastAPI backend with a React frontend would handle load properly.

---

## 📁 Project Structure

```
shelfscan/
│
├── app.py                      # Streamlit entry point — orchestrates everything
├── pyproject.toml              # Dependencies managed by uv
├── packages.txt                # System deps for Streamlit Cloud (libzbar0, tesseract-ocr)
├── .env                        # Secrets — DATABASE_URL, JWT_SECRET (never committed)
│
├── ml/
│   ├── health_model.py         # GradientBoostingRegressor health scorer, 0–100
│   ├── image_classifier.py     # OpenCV barcode region detection + CLIP pipeline
│   └── price_intelligence.py   # Health-per-penny and nutrition density calculator
│
├── utils/
│   ├── api.py                  # Open Food Facts API wrapper with caching
│   ├── barcode.py              # 5-stage barcode decoder
│   ├── session.py              # Streamlit session state management
│   ├── auth.py                 # bcrypt signup/login + JWT verify
│   └── db.py                   # SQLite (dev) / PostgreSQL (prod) CRUD
│
├── components/
│   ├── styles.py               # CSS injected via st.markdown
│   └── ui.py                   # render_hero(), render_product_card(), render_compare_panel()
│
└── tests/
    ├── test_barcode.py         # CV pipeline tests with real barcode images
    ├── test_api.py             # API tests with mocked requests
    ├── test_health_model.py    # ML sanity checks (Nutella should score < Evian)
    └── test_price.py           # Price intelligence output range tests
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **CV / Vision** | OpenCV 4.13 | Barcode region detection, CLAHE, morphological ops |
| **Barcode** | pyzbar 0.1.9 | Primary EAN-13/UPC-A decoder, wraps libzbar C library |
| **Barcode** | zxing-cpp 3.0 | Stage 5 fallback — different algorithm, different failure modes |
| **OCR** | pytesseract 0.3 | Text extraction fallback for damaged barcodes |
| **Images** | Pillow 10.4 | All image I/O before the CV pipeline |
| **ML** | scikit-learn 1.8 | GradientBoostingRegressor, StandardScaler, Pipeline |
| **Numerics** | numpy 2.4 | Feature vectors, image arrays, K-Means colour clustering |
| **Data** | pandas 2.2 | Product comparison tables, scan history, nutrient cleaning |
| **Deep Learning** | PyTorch 2.10 (CPU) | CLIP inference backend |
| **Image AI** | transformers 5.2 | HuggingFace CLIP — zero-shot product image classification |
| **API** | requests 2.32 | Open Food Facts HTTP calls |
| **Data Source** | Open Food Facts | 3M+ products, free, no API key, returns Nutri-Score + NOVA |
| **Auth** | bcrypt 5.0 | Password hashing — deliberately slow to resist brute force |
| **Auth** | PyJWT 2.11 | Stateless JWT tokens — no server-side session storage needed |
| **Database** | SQLite → PostgreSQL | Dev: zero setup. Prod: handles concurrent users |
| **UI** | Streamlit 1.40 | Full web UI — camera on mobile, tabs, score bars, compare mode |
| **Testing** | pytest | Unit tests across all four modules |

---

## 🚀 Run It Locally

### Prerequisites

**Fedora:**
```bash
sudo dnf install zbar zbar-devel tesseract tesseract-devel
```

**Ubuntu / Debian:**
```bash
sudo apt-get install libzbar0 tesseract-ocr libtesseract-dev
```

**macOS:**
```bash
brew install zbar tesseract
```

### Install & Run

```bash
git clone https://github.com/codershanks/shelfscan.git
cd shelfscan

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install all dependencies
uv sync

# Verify
python -c "import cv2, pyzbar, sklearn, numpy, pandas, PIL, torch, transformers; print('✅ All imports OK')"

# Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). On mobile, the file uploader becomes a camera button automatically.

### Environment Variables

```
DATABASE_URL=sqlite:///shelfscan.db
JWT_SECRET=your-secret-key-here
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🗺️ How I Build it

The hackathon version demonstrates the full pipeline end-to-end. Production additions planned:

- [ ] **Live store prices** — Tesco Developer API for real-time pricing
- [ ] **Retailer price comparison** — Open Grocery API across multiple stores
- [ ] **Price drop alerts** — SendGrid email notifications
- [ ] **PostgreSQL** — production multi-user database
- [ ] **FastAPI backend** — decouple ML inference from UI for scalability
- [ ] **Personalised scoring** — dietary profile adjustments (vegan, allergen-aware, diabetic)
- [ ] **Mobile app** — React Native wrapper with native camera access

---

## 🤔 Questions Judges Ask

**Why not just show Nutri-Score directly?**

Nutri-Score is a good signal but incomplete. It doesn't penalise ultra-processed formulation (NOVA group 4), doesn't account for additive count, and doesn't reward organic certification. The GBM model combines Nutri-Score with NOVA, additive count, and 7 raw nutrient values to produce a more complete picture — and logs every adjustment so users can see and disagree with any decision.

**What if the barcode isn't in the Open Food Facts database?**

The app shows a not-found message with a direct link to add the product. In production, pytesseract reads the product name and quantity directly from the label text and falls back to a text search.

**How would you scale this to production?**

Swap SQLite for PostgreSQL. Add a Redis cache layer in front of Open Food Facts API calls. Host the ML models as a separate FastAPI service so inference scales independently of the UI. Use Celery for async tasks like price alerts.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

Built by **codershanks** · Meta × MLH Hackathon 2025

Python · OpenCV · pyzbar · scikit-learn · CLIP · Open Food Facts · Streamlit

</div>