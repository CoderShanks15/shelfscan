<div align="center">

#  🛒 ShelfScan
### *Health Intelligence at the Speed of Sight *

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)

**Built for the Meta × MLH Hackathon 2025**

</div>

---

## 💡 The Vision: Why I Built This

Every day, millions of people stand in supermarket aisles holding two products, paralyzed by choice. They want to eat healthier, but they are met with a wall of confusing nutrition labels, marketing buzzwords, and hidden additives. **Health information should not require a degree in nutrition to decode.**

I built **ShelfScan** to solve this. It is not a search engine; it is an instant verdict.

Point your camera at a barcode. Under the hood, a **5-stage computer vision pipeline** decodes the image, a **machine learning model** calculates a health score (0-100), an **AI vision model** (CLIP) categorizes the packaging, and a **Price Intelligence engine** calculates the raw nutritional value per penny. 

*All in under 3 seconds.*

---

## 🧠 The Technical Depth

ShelfScan isn't just an API wrapper; it's a multi-layered intelligence engine built to handle messy real-world data.

### 1. Robust 5-Stage Computer Vision Pipeline
Real-world photos are blurry, angled, and poorly lit. To guarantee a reliable scan, I built a cascade decoder:
*   **Stage 1:** Direct structural decoding (`pyzbar`)
*   **Stage 2:** Multi-angle rotational scans for tilted products
*   **Stage 3:** CLAHE contrast preprocessing for dark/shadowed images
*   **Stage 4:** Adaptive thresholding for uneven lighting
*   **Stage 5:** Algorithmic fallback (`zxing-cpp`) for structurally stubborn barcodes

### 2. Explainable AI Scoring (Gradient Boosting)
Why use a Gradient Boosting Machine (GBM) instead of a black-box Neural Network? **Trust.** 
When ShelfScan scores a product a `28/100`, it explains exactly why: *"−12 points for 8 additives, −8 for ultra-processed NOVA group, +3 for high fibre."* The ML model uses 9 distinct nutritional vectors to calculate its score, offering transparency that a standard neural net cannot.

### 3. Zero-Shot Image Classification
I integrated OpenAI's **CLIP** (`clip-vit-base`) to visually classify the product category directly from your uploaded photo. By leveraging Contrastive Language–Image Pretraining, ShelfScan categorizes images (e.g., "fresh produce", "packaged goods") with zero fine-tuning and zero custom labeling needed.

### 4. Premium, High-Performance UI/UX
The application features a bespoke, "Linear-style" **Dark Mode Design System**:
*   **Custom Motion Graphics:** A pure, highly performant CSS mesh gradient background that renders floating, glowing orbs of teal and purple without relying on external WebGL libraries.
*   **Curved Aesthetics:** 100px pill-shaped buttons, unified dropzones, and seamless transition animations that make a Python dashboard feel like a world-class SaaS product.

---

## 🤖 Models Used & Availability

ShelfScan relies on three AI/ML models, all of which are **open-source and free to use indefinitely** — there are no API keys, subscriptions, or usage quotas required.

| # | Model | Purpose | License | Usage Limit |
|---|-------|---------|---------|-------------|
| 1 | **LightGBM Regressor** | Calculates the health score (0–100) from 150 nutritional features | [MIT](https://github.com/microsoft/LightGBM/blob/master/LICENSE) | **Unlimited** — trained and runs entirely on your local machine |
| 2 | **CLIP ViT-Base-Patch32** (`openai/clip-vit-base-patch32`) | Zero-shot image classification of product packaging categories | [MIT](https://github.com/openai/CLIP/blob/main/LICENSE) | **Unlimited** — downloaded once (~340 MB) from HuggingFace and runs locally thereafter |
| 3 | **OpenCV** | Image quality analysis, barcode region detection, and colour extraction | [Apache 2.0](https://github.com/opencv/opencv/blob/master/LICENSE) | **Unlimited** — open-source library bundled with the project |

> **Summary:** All three models are open-source with permissive licenses. None of them have time limits, rate limits, or require a paid plan. You can use them for as long as you like.

---

## 🎯 See It in Action

Try these barcodes in the **Scan** tab right now (no camera required):

| Product | Barcode | Score | Verdict |
|---|---|---|---|
| **Nutella 400g** | `3017620422003` | ~28 | ⚠️ Caution |
| **Pringles Original** | `5053990101538` | ~22 | ❌ Avoid |
| **Quaker Oats 1kg** | `5000173008065` | ~74 | ✅ Great |
| **Evian Water 1.5L** | `3068320113994` | ~88 | ✅ Great |

*Pro Tip: Scan Nutella, then Evian Water, and go to the **Compare** tab to see them face-off side-by-side.*

---

## 🚀 Run It Locally

### 1. System Prerequisites
You need native libraries for the CV pipeline to run:
*   **macOS:** `brew install zbar tesseract`
*   **Ubuntu/Debian:** `sudo apt-get install libzbar0 tesseract-ocr libtesseract-dev`
*   **Fedora:** `sudo dnf install zbar zbar-devel tesseract tesseract-devel`

### 2. Install & Run
```bash
git clone https://github.com/codershanks/shelfscan.git
cd shelfscan

# Set up environment and dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync

# Run the Intelligence Engine
streamlit run app.py
```
*Note: On mobile devices, the file uploader will automatically request Camera access for live scanning.*

---

<div align="center">
Built with passion by <b>codershanks</b> for the <b>Meta × MLH Hackathon 2025</b>.
</div>