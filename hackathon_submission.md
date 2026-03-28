# 🛒 ShelfScan - Meta × MLH Hackathon 2025 Submission

## 💡 Inspiration
Every day, millions of people stand in supermarket aisles holding two products, paralyzed by choice. They want to eat healthier but are met with a wall of confusing nutrition labels, marketing buzzwords, and hidden additives. **Health information should not require a degree in nutrition to decode.** We wanted to solve this problem by providing an instant, clear, and actionable verdict on any food product.

## 🚀 What it does
ShelfScan is a multi-layered health intelligence engine, not just a search tool. By pointing a camera at a barcode, ShelfScan delivers an instant verdict:
- **Lightning Fast Decoding:** Processes the product in under 3 seconds.
- **Explainable Health Score (0-100):** Unlike black-box models, ShelfScan details exactly *why* points were added or deducted (e.g., "−12 points for 8 additives, −8 for ultra-processed NOVA group, +3 for high fibre").
- **Zero-Shot Visual Categorization:** Uses AI vision models to look at the product packaging and categorize it automatically.
- **Price Intelligence:** Calculates the raw nutritional value per penny to help users optimize both their health and their budget.

## 🛠️ How we built it
We built a robust, locally runnable web application using **Python, Streamlit, scikit-learn, and OpenCV**.

1. **5-Stage Computer Vision Pipeline:** Real-world shopping photos are messy—they can be blurry, angled, or poorly lit. We built a custom cascade decoder using `pyzbar` and `zxing-cpp` that applies multi-angle rotational scans, CLAHE contrast preprocessing, and adaptive thresholding to ensure reliable scans every time.
2. **Explainable AI Scoring (Gradient Boosting):** We prioritized user trust by implementing a Gradient Boosting Machine (GBM) that evaluates 9 distinct nutritional vectors. This allows us to provide full transparency into the final score.
3. **Zero-Shot Image Classification:** We integrated OpenAI's **CLIP** (`clip-vit-base`) model through Contrastive Language–Image Pretraining to categorize products instantly without any custom fine-tuning or labeling.
4. **Premium UI/UX System:** We bypassed the standard look of data dashboards to build a "Linear-style" Dark Mode app. By injecting bespoke CSS, we created a highly performant mesh gradient background with floating orbs, pill-shaped buttons, and seamless transition animations—making a Python script feel like a world-class SaaS product.

## ⚠️ Challenges we ran into
Building a reliable barcode scanner that works under supermarket lighting with glare and awkward camera angles was exceptionally difficult. Standard decoders failed frequently. We overcame this by engineering a custom 5-stage vision pipeline that dynamically applies preprocessing techniques (like CLAHE) and structural fallbacks only when necessary, optimizing both accuracy and speed.

## 🏆 Accomplishments that we're proud of
- Achieving an **under 3-second end-to-end pipeline** from image capture to an explainable AI verdict.
- Implementing a truly **explainable ML model** that empowers users with transparent nutritional insights instead of opaque scores.
- Elevating the UI of a Streamlit app to look and behave like a premium, custom-built React/Next.js application using advanced CSS motion graphics.

## 📚 What we learned
- The extensive level of image preprocessing required to make computer vision reliable in real-world, uncontrolled environments.
- How to seamlessly unify multi-modal AI (CLIP for vision, GBM for tabular data) into a single, cohesive, and fast inference pipeline.
- Advanced frontend engineering within Streamlit, pushing the boundaries of what is possible with raw CSS injections to completely transform the user experience.

## ⏭️ What's next for ShelfScan
- **Native Mobile Experience:** Transitioning the core pipeline into a native iOS/Android app to enable even faster, on-the-go scanning right in the aisle.
- **Personalized Health Profiles:** Adapting the scoring engine to account for user-specific dietary needs, allergies, and goals (e.g., diabetic-friendly, high-protein, vegan).
- **Gamification & Tracking:** Allowing users to track their scanned product history and earn rewards for consistently choosing healthier alternatives over time.
