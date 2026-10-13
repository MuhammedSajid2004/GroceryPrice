# 🛒 Costco Grocery Price Prediction & Analytics App

An end-to-end, production-grade Machine Learning application and interactive Streamlit web interface designed to predict grocery prices, estimate promotional impact, and benchmark price points across Costco wholesale categories.

Built from a scraped Costco catalog dataset (`GroceryDataset.csv` / `costco_grocery.csv`).

---

## 📌 Table of Contents
1. [Project Overview & Business Value](#-project-overview--business-value)
2. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
3. [Step 1: Data Exploration & Cleaning](#-step-1-data-exploration--cleaning)
4. [Step 2: Feature Engineering & Extraction](#-step-2-feature-engineering--extraction)
5. [Step 3: Modeling & Cross-Validation](#-step-3-modeling--cross-validation)
6. [Step 4: Data Leakage Check (Crucial ML Engineering)](#-step-4-data-leakage-check-crucial-ml-engineering)
7. [Step 5: Model Performance Benchmarks](#-step-5-model-performance-benchmarks)
8. [Step 6: Streamlit Web Application](#-step-6-streamlit-web-application)
9. [Setup & Quickstart Guide](#-setup--quickstart-guide)
10. [Model Limitations & Real-World Considerations](#-model-limitations--real-world-considerations)

---

## 🎯 Project Overview & Business Value

Wholesale grocery retailers like Costco operate with a distinct pricing dynamic characterized by bulk multipacks, commercial quantities, luxury wholesale items (such as caviar kilos and whole Parmigiano Reggiano wheels), and intermittent manufacturer promotions.

This application provides:
1. **Accurate Base Price Estimation:** Estimates the fair market baseline price of any product directly from text attributes, category, weight, and pack quantity.
2. **Promotional Discount Impact:** Applies dollar (`$`) or percentage (`%`) discounts dynamically post-inference, avoiding target leakage while producing accurate member checkout prices.
3. **Uncertainty Calibration:** Generates 80% and 90% empirical prediction intervals calibrated using validation residual percentiles.
4. **Category Benchmarking:** Evaluates where the predicted item sits within its subcategory distribution (Budget, Mid-tier, Premium/Bulk).

---

## 📂 Project Architecture & Directory Structure

```text
store/
├── data/
│   └── costco_grocery.csv             # Scraped Costco grocery dataset (1,757 rows)
├── notebooks/
│   ├── costco_price_prediction.ipynb  # Interactive Jupyter walkthrough
│   ├── train_eda_notebook.py          # Standalone commented exploratory script
│   ├── generate_eda_plots.py          # Script generating high-resolution EDA figures
│   └── eda_plots/                     # Exported EDA visualizations
├── src/
│   ├── __init__.py                    # Python package initializer
│   ├── preprocessing.py               # Custom scikit-learn transformers & pipeline builders
│   ├── train.py                       # Automated training, 5-fold CV, Optuna tuning & artifact export
│   └── predict.py                     # Production inference engine with confidence intervals
├── artifacts/
│   ├── model.joblib                   # Serialized production pipeline
│   ├── metadata.json                  # Model hyperparams, test metrics, and category statistics
│   └── plots/                         # Feature importance and diagnostic plots
├── app.py                             # Full-featured Streamlit web application
├── requirements.txt                   # Pinned dependency requirements
└── README.md                          # Comprehensive engineering documentation
```

---

## 🔍 Step 1: Data Exploration & Cleaning

### Key Dataset Statistics
- **Total Initial Rows:** 1,757 entries across 8 columns.
- **Columns:** `Sub Category`, `Price`, `Discount`, `Rating`, `Title`, `Currency`, `Feature`, `Product Description`.

### Data Cleaning Decisions & Justifications
1. **Price Parsing & Range Resolution:**
   - Raw prices were string-formatted with `$` and commas (e.g. `$56.99 `).
   - 14 products contained price ranges (e.g. `$32.99through-$83.99` for simplehuman trash can liners of multiple sizes). These were resolved by calculating the **midpoint** of the range.
   - 3 rows had null prices (warehouse-only items with no online price); these were dropped.
   - 2 scraper error pages (`Title: "Page Not Found!"`) were detected and pruned.
   - Final modeled dataset: **1,752 clean rows**.
2. **Currency Verification:**
   - `Currency` had only one unique value: `$` (USD) with 5 NaNs. Dropping or treating currency as constant is 100% justified.
3. **Customer Rating Handling:**
   - 61.2% of rows had missing ratings or `'No Reviews'`.
   - Rather than dropping or arbitrary imputation, we extracted `rating_value`, `review_count`, and created an explicit boolean indicator: `rating_is_missing`. Missing ratings were then imputed with the median (4.5). In wholesale environments, the *absence* of customer reviews strongly signals commercial institutional supply rather than poor quality.
4. **Outlier Treatment & Log Transformation:**
   - Raw prices range from **$3.99 to $1,999.99** with a median of **$23.99** and mean of **$61.34**.
   - Standard IQR outlier fences ($Q3 + 1.5 \times \text{IQR} = \$109.99$) flag 238 items (13.6% of data) as outliers.
   - Inspection revealed these are **genuine luxury wholesale items** (e.g., *Plaza Golden Osetra Caviar Kilo Pack* at $1,999.99, *Whole Wheel Parmigiano Reggiano 72 lbs* at $949.99, *Japanese A5 Wagyu Tenderloin Roast* at $999.99).
   - **Decision:** Do NOT silently drop luxury items. Instead, we train on the target $\log(1 + \text{Price})$:
     $$\tilde{y} = \log(1 + y)$$
     This compresses the severe right tail into an approximately bell-shaped distribution, stabilizing gradient updates across all models. Predictions are converted back using $\exp(\tilde{y}) - 1$.

---

## 🛠️ Step 2: Feature Engineering & Extraction

Preprocessing is packaged into reusable scikit-learn transformers:

1. **Physical Unit & Quantity Regex Extractor (`UnitFeatureExtractor`):**
   - Extracts weight in ounces (`weight_oz`), normalizing from pounds (`lbs`), kilograms (`kg`), grams (`g`), and fluid ounces (`fl oz`).
   - Extracts item count (`item_count`), capturing terms like `48 count`, `100 ct`, `14 servings`, `158 loads`.
   - Extracts pack count (`pack_count`), capturing `2-pack`, `6-pack`, `12-pack`.
   - Extracts text length, title length, and bulk/organic flags.
2. **Text Representation:**
   - Concatenates `Title + " " + Feature + " " + Product Description`.
   - Baseline: `TfidfVectorizer` (1–2 ngrams, `max_features=4000`, `min_df=2`, `stop_words='english'`).
   - Latent Semantic Analysis (`TruncatedSVD` with 50 components): Dense semantic projection capturing vocabulary nuances without requiring multi-gigabyte PyTorch transformer weights, ensuring sub-50ms web inference.
3. **Categorical Encoding:**
   - `OneHotEncoder(handle_unknown='ignore')` for the 19 distinct grocery subcategories.

---

## 🛡️ Step 4: Data Leakage Check (Crucial ML Engineering)

A critical data leakage consideration arises from the `Discount` column:
- In the scraped data, discounts appear as `"After $5 OFF"` or `"After $70 OFF"`.
- Only **7.2% of items (126 of 1,752)** carry promotional discounts.

### Leakage Analysis:
1. **Target-Discount Confounding:** Promotional discounts are external marketing decisions applied *after* base pricing is determined. Feeding discount amounts as input features to predict product price creates a circular causal dependency.
2. **Inference Failure:** When predicting the price of a standard catalog item, the discount is unknown. A model trained on discount features learns spurious correlations between discount size and item tier.

### Empirical Validation:
We trained two distinct pipeline variants across identical 80/10/10 splits:
- **Variant A (With Discount Features):** Test $R^2 = 0.8661$, 5-Fold CV $R^2 = 0.8443$.
- **Variant B (Without Discount Features - Selected):** Test $R^2 = 0.8655$, 5-Fold CV $R^2 = 0.8443$.

**Conclusion:** Variant B matches Variant A's predictive accuracy while guaranteeing zero leakage. In production:
$$\text{Base Price} = \hat{f}(\text{Title}, \text{Sub Category}, \text{Description}, \text{Units}, \text{Rating})$$
$$\text{Final Price} = \max(0.50, \ \text{Base Price} - \text{Discount})$$

---

## 📊 Step 5: Model Performance Benchmarks

All models were evaluated on the held-out 10% test set (176 unseen items). Metrics are reported on **original dollar units ($)** alongside log-space $R^2$:

| Model | Variant | 5-Fold CV $R^2$ (log) | Test MAE ($) | Test RMSE ($) | Test MAPE (%) | Test $R^2$ (log) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Tuned)** | **Variant B** | **0.8994** | **$23.00** | **$91.76** | **23.52%** | **0.8654** |
| Random Forest (Base) | Variant B | 0.8443 | $23.12 | $93.75 | 23.73% | 0.8655 |
| Gradient Boosting | Variant B | 0.8462 | $24.71 | $98.12 | 24.46% | 0.8535 |
| Ridge Regression | Variant B | 0.8173 | $25.29 | $107.03 | 27.10% | 0.8248 |
| LightGBM | Variant B | 0.8247 | $25.96 | $109.15 | 27.07% | 0.8186 |
| XGBoost | Variant B | 0.8378 | $26.11 | $109.28 | 26.96% | 0.8183 |
| Random Forest | Variant A (Leak) | 0.8443 | $23.20 | $93.58 | 23.82% | 0.8661 |
| Gradient Boosting | Variant A (Leak) | 0.8508 | $24.58 | $98.80 | 24.30% | 0.8513 |

### Hyperparameter Tuning (Optuna)
- **Framework:** Optuna Tree-structured Parzen Estimator (TPE) sampler.
- **Optimal Parameters:** `n_estimators=160`, `max_depth=24`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features=0.468`, `n_components=50`.
- **Top Feature Drivers:**
  1. `Rating_Missing` & `Review_Count` (Institutional/bulk signal)
  2. `Text_Semantic_SVD_5` & `Text_Semantic_SVD_2` (Luxury & bulk text semantics)
  3. `Sub Category_Meat & Seafood` (High base asset value)
  4. `Text_Length` & `Title_Length` (Product specification richness)
  5. `Weight_oz` & `Item_Count` (Quantity scaling)

---

## 💻 Step 6: Streamlit Web Application

The Streamlit app (`app.py`) provides an interactive interface:
- **Interactive Product Input:** Title, Category dropdown (19 categories), customer rating slider, unrated checkbox, bullet points, and product description.
- **Quick-Fill Templates:** One-click pre-fills for Olive Oil, Salmon Case, Prebiotic Soda, and Caviar.
- **Promotion Settings:** Toggle between Dollar (`$`) and Percentage (`%`) discounts.
- **Live Metric Displays:** Base Price, Applied Discount, and Final Member Price.
- **Uncertainty Bounds:** 80% and 90% confidence intervals.
- **Category Benchmarking Chart:** Interactive Plotly boxplot visualizing the item's position relative to its category median and quartiles.
- **Model Insights & EDA Tabs:** Live metrics table, feature importance visualization, and exploratory charts.

---

## 🚀 Setup & Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.13)

### 2. Environment Setup
```powershell
# Clone or navigate to the project directory
cd "c:\Users\Muhammed Sajid\Documents\Projects\store"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Training (End-to-End Pipeline)
```powershell
python src/train.py
```
*Executes data cleaning, runs 5-fold CV across 5 algorithms, validates leakage variants, performs Optuna tuning, and writes `artifacts/model.joblib` and `artifacts/metadata.json`.*

### 4. Launch the Streamlit App
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run Standalone Inference
```powershell
python src/predict.py
```

---

## ⚠️ Model Limitations & Real-World Considerations

1. **Catalog Regionality:** Costco prices can vary between warehouse locations (e.g. Hawaii/Alaska vs contiguous US) and online delivery fees ($3.00/item delivery markup). The dataset reflects online delivery catalog pricing.
2. **Text Representation Scalability:** TF-IDF with TruncatedSVD was selected over heavy transformer embeddings (`all-MiniLM-L6-v2`) to maintain sub-50ms inference latency without requiring 2.5GB PyTorch GPU runtimes. For production workloads with multi-language catalogs, fine-tuned dense embeddings can be swapped into `src/preprocessing.py`.
3. **Extreme Luxury Long-Tail:** While log-transformation successfully prevents catastrophic loss gradients on $1,999.99 caviar, high-end commercial roasts have higher absolute dollar variance than everyday staples. The included residual percentiles communicate this uncertainty directly to users.
#   G r o c e r y P r i c e  
 