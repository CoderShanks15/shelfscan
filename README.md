# ShelfScan 🛒

> **3-second health verdict on any supermarket product. Snap a barcode → ML scores it → you decide.**

Built for the **Meta × MLH Hackathon 2025**

---

## The Problem

Standing in a supermarket aisle comparing two similar products at different prices — you can't tell which is healthier, better value, or worth paying more for. Google gives you search results. ShelfScan gives you a direct verdict backed by computer vision, machine learning, and real product data.

## The Solution

Snap a photo of any product barcode → 5-stage CV pipeline decodes it → ML model scores health 0–100 → CLIP classifies the product image → price intelligence calculates health-per-penny → full breakdown shown in 3 seconds.

---

## Demo

### Try these barcodes instantly

| Product | Barcode | Expected Score | Verdict |
|---|---|---|---|
| Nutella 400g | `3017620422003` | ~28 | ⚠️ Caution |
| Haribo Gold-Bears | `4001686323564` | ~32 | ⚠️ Caution |
| Kellogg's Corn Flakes | `5010477348549` | ~45 | 🟡 OK |
| Evian Water 1.5L | `3068320113994` | ~88 | ✅ Great |
| Pringles Original | `5053990101538` | ~22 | ❌ Avoid |
| Quaker Oats 1kg | `5000173008065` | ~74 | ✅ Great |

---

## Tech Stack

### Computer Vision & Barcode Decoding
- **OpenCV** — barcode region detection via morphological ops, CLAHE preprocessing for dark photos
- **pyzbar** — primary barcode decoder (wraps libzbar C library)
- **pytesseract** — OCR fallback when barcode is damaged or absent
- **zxing-cpp** — Stage 5 fallback decoder, catches barcodes pyzbar misses
- **Pillow** — all image I/O before the CV pipeline

### Machine Learning
- **scikit-learn** — GradientBoostingRegressor health scorer, StandardScaler, Pipeline
- **numpy** — feature matrix construction, image array ops
- **pandas** — product comparison, scan history, nutrient data cleaning
- **transformers + torch** — CLIP zero-shot image classification (no training data needed)
- **torchvision** — image preprocessing for CLIP inference

### Data
- **Open Food Facts API** — 3M+ product database, free, no API key required
- **requests** — HTTP calls with caching via `@st.cache_data`

### Auth & Database
- **bcrypt** — password hashing
- **PyJWT** — stateless auth tokens
- **SQLite** (dev) / **PostgreSQL** (prod) — user accounts, scan history, favourites

### UI & Deployment
- **Streamlit** — entire web UI, works as camera button on mobile
- **pytest** — unit tests for each module

---

## How It Works

```
User uploads photo or types barcode
            ↓
OpenCV detects barcode region (bounding box)
            ↓
5-stage cascade: pyzbar → rotation → CLAHE → adaptive threshold → zxing-cpp
            ↓
EAN-13 / UPC-A string  e.g. 3017620422003
            ↓
Open Food Facts API → full product JSON (600+ fields)
            ↓
9 features extracted: energy, fat, saturated fat, sugar, fiber, protein, salt, nova, additives
            ↓
GradientBoosting model → raw score 0–100
            ↓
Rule-based adjustments logged (organic +5, ultra-processed −15, etc.)
            ↓
CLIP classifies product image zero-shot → category + confidence
            ↓
Price intelligence → health-per-penny score
            ↓
Streamlit renders full breakdown with tabs, bars, and compare mode
```

---

## Project Structure

```
shelfscan/
├── app.py                      # Main Streamlit entry point
├── pyproject.toml              # Dependencies (managed by uv)
├── uv.lock                     # Locked dependency versions
├── packages.txt                # System deps for Streamlit Cloud
├── .env                        # Secrets — never committed
├── README.md
│
├── ml/
│   ├── health_model.py         # GBM health scorer (0–100)
│   ├── image_classifier.py     # OpenCV + CLIP pipeline
│   └── price_intelligence.py   # Health-per-penny calculator
│
├── utils/
│   ├── api.py                  # Open Food Facts API wrapper
│   ├── barcode.py              # 5-stage barcode decoder
│   ├── session.py              # Streamlit session state
│   ├── auth.py                 # signup, login, JWT verify
│   └── db.py                   # SQLite/PostgreSQL CRUD
│
├── components/
│   ├── styles.py               # CSS injected via st.markdown
│   └── ui.py                   # Reusable UI components
│
└── tests/
    ├── test_barcode.py
    ├── test_api.py
    ├── test_health_model.py
    └── test_price.py
```

---

## Setup & Installation

### Prerequisites

**Fedora:**
```bash
sudo dnf install zbar zbar-devel tesseract tesseract-devel
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libzbar0 tesseract-ocr libtesseract-dev
```

**macOS:**
```bash
brew install zbar tesseract
```

### Install

```bash
# Clone the repo
git clone https://github.com/codershanks/shelfscan.git
cd shelfscan

# Create and activate virtual environment (using uv)
uv venv .venv --python 3.12
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

# Install all dependencies
uv sync

# Verify everything is working
python -c "import cv2, pyzbar, sklearn, numpy, pandas, PIL, torch, transformers; print('✅ All imports OK')"
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment Variables

Create a `.env` file in the project root:

```
DATABASE_URL=sqlite:///shelfscan.db
JWT_SECRET=your-secret-key-here
```

---

## Running Tests

```bash
pytest tests/
```

---

## Deployment

This app is configured for **Streamlit Cloud** deployment.

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Deploy

The `packages.txt` file handles system dependency installation on Streamlit Cloud automatically.

---

## Why Gradient Boosting and not a neural network?

GBM gives interpretability. Every feature's contribution to the score is logged and shown in the UI — *"your score dropped 12 points because of 8 additives, gained 5 for organic certification."* A neural network would be a black box. For a health tool, explainability matters more than marginal accuracy gains.

## How does CLIP work without training data?

CLIP was trained by OpenAI on 400 million image-text pairs from the internet. It learns a shared embedding space where images and text that describe the same thing end up close together. ShelfScan gives it a product photo and a list of text labels — *"fresh produce"*, *"ultra-processed packaging"* — and it scores which text best matches the image. Zero training data required.

---

## Roadmap

- [ ] Live store price integration (Tesco Developer API)
- [ ] Price comparison across retailers (Open Grocery API)
- [ ] Price drop alerts via email (SendGrid)
- [ ] PostgreSQL for production multi-user support
- [ ] FastAPI backend for high-concurrency deployment
- [ ] Mobile app wrapper

---

## Built With

Python · OpenCV · pyzbar · scikit-learn · CLIP · Open Food Facts · Streamlit

**Meta × MLH Hackathon 2025**