
# 🛒 Costco Grocery Price Prediction & Analytics App

An end-to-end Machine Learning application with an interactive Streamlit interface designed to predict grocery prices, estimate promotional discounts, and benchmark prices across Costco wholesale categories.

Built using the scraped Costco catalog dataset (`GroceryDataset.csv` / `costco_grocery.csv`).

---

## 📌 Table of Contents

1. [Project Overview & Business Value](#-project-overview--business-value)
2. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
3. [Step 1: Data Exploration & Cleaning](#-step-1-data-exploration--cleaning)
4. [Step 2: Feature Engineering & Extraction](#-step-2-feature-engineering--extraction)
5. [Step 3: Modeling & Cross-Validation](#-step-3-modeling--cross-validation)
6. [Step 4: Data Leakage Check](#-step-4-data-leakage-check-crucial-ml-engineering)
7. [Step 5: Model Performance Benchmarks](#-step-5-model-performance-benchmarks)
8. [Step 6: Streamlit Web Application](#-step-6-streamlit-web-application)
9. [Setup & Quickstart Guide](#-setup--quickstart-guide)
10. [Model Limitations & Real-World Considerations](#-model-limitations--real-world-considerations)

---

## 🎯 Project Overview & Business Value

Wholesale grocery retailers like Costco have distinctive pricing patterns involving bulk multipacks, commercial quantities, premium wholesale products, and occasional manufacturer promotions.

This application provides:

1. **Base Price Estimation:** Predicts a product's baseline price using its title, category, description, weight, pack quantity, and other attributes.
2. **Promotional Discount Calculation:** Applies dollar (`$`) or percentage (`%`) discounts after price prediction to calculate the final member price.
3. **Prediction Intervals:** Generates 80% and 90% empirical prediction intervals using validation residual percentiles.
4. **Category Benchmarking:** Compares predicted prices with category distributions to classify products as Budget, Mid-tier, or Premium/Bulk.

---

## 📂 Project Architecture & Directory Structure

```text
store/
├── data/
│   └── costco_grocery.csv
├── notebooks/
│   ├── costco_price_prediction.ipynb
│   ├── train_eda_notebook.py
│   ├── generate_eda_plots.py
│   └── eda_plots/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── artifacts/
│   ├── model.joblib
│   ├── metadata.json
│   └── plots/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔍 Step 1: Data Exploration & Cleaning

### Key Dataset Statistics

- **Initial Rows:** 1,757
- **Original Columns:** 8
- **Final Cleaned Dataset:** 1,752 rows
- **Columns:** `Sub Category`, `Price`, `Discount`, `Rating`, `Title`, `Currency`, `Feature`, `Product Description`

### Data Cleaning Decisions

1. **Price Parsing & Range Resolution**
   - Converted string-formatted prices containing currency symbols and commas into numeric values.
   - Resolved 14 price ranges by calculating their midpoints.
   - Removed 3 rows with missing prices.
   - Removed 2 scraper error pages identified by the title `Page Not Found!`.

2. **Currency Verification**
   - The `Currency` column contained only USD (`$`) values, with 5 missing entries.
   - Currency was treated as a constant rather than a predictive feature.

3. **Customer Rating Handling**
   - Approximately 61.2% of rows had missing ratings or `No Reviews`.
   - Extracted `rating_value` and `review_count`.
   - Created a `rating_is_missing` indicator and imputed missing numerical ratings using the median of 4.5.

4. **Outlier Analysis & Log Transformation**
   - Minimum price: $3.99
   - Maximum price: $1,999.99
   - Median price: $23.99
   - Mean price: $61.34
   - The IQR method identified 238 potential outliers.

   High-priced products were retained because they represented legitimate premium and bulk products.

   The target variable was transformed using:

   \[
   \tilde{y} = \log(1+y)
   \]

   Predictions are converted back to the original price scale using:

   \[
   \hat{y} = \exp(\hat{\tilde{y}})-1
   \]

---

## 🛠️ Step 2: Feature Engineering & Extraction

Preprocessing is organized into reusable scikit-learn transformers and pipeline components.

### 1. Physical Unit & Quantity Extraction

The custom `UnitFeatureExtractor` extracts:

- `weight_oz`: Weight normalized to ounces.
- `item_count`: Quantities such as servings and loads.
- `pack_count`: Multipack quantities such as 2-pack and 12-pack.
- Text length and title length.
- Bulk and organic product indicators.

### 2. Text Representation

The following fields are concatenated:

`Title + Feature + Product Description`

- **TF-IDF:** Unigrams and bigrams with `max_features=4000`, `min_df=2`, and English stop-word removal.
- **Truncated SVD:** Reduces the TF-IDF representation to 50 components.

### 3. Categorical Encoding

`OneHotEncoder(handle_unknown='ignore')` encodes product subcategories and supports previously unseen categories during inference.

---

## 🧠 Step 3: Modeling & Cross-Validation

The training pipeline compares multiple regression algorithms to identify a suitable model for grocery price prediction.

### Models Evaluated

- Random Forest Regression
- Gradient Boosting Regression
- Ridge Regression
- LightGBM
- XGBoost

### Training Workflow

1. Load and clean the dataset.
2. Extract numerical, categorical, and text-based features.
3. Transform target prices using `log1p`.
4. Split the data into training, validation, and test sets.
5. Perform five-fold cross-validation.
6. Compare models using MAE, RMSE, MAPE, and R².
7. Tune Random Forest hyperparameters using Optuna.
8. Evaluate the selected model on the held-out test set.
9. Export the trained model and metadata.

All learned preprocessing operations should be fitted on training data only to avoid data leakage.

---

## 🛡️ Step 4: Data Leakage Check (Crucial ML Engineering)

The `Discount` column requires careful treatment because promotional discounts may not be known when predicting a new product's baseline price.

Examples include `After $5 OFF` and `After $70 OFF`.

Only 7.2% of the cleaned dataset (126 of 1,752 products) reportedly contains promotional discounts.

### Leakage Analysis

1. **Target-Discount Confounding:** Discount amounts may introduce misleading relationships between promotions and product prices.
2. **Inference Limitations:** A new product's discount may be unknown during prediction.
3. **Feature Consistency:** Training and inference must use consistent feature definitions.

### Empirical Validation

Two variants were evaluated using the same data split.

| Variant | Discount Features | Test R² | 5-Fold CV R² |
|---|---|---:|---:|
| Variant A | Included | 0.8661 | 0.8443 |
| Variant B | Excluded | 0.8655 | 0.8443 |

**Selected approach:** Variant B, which excludes discount information from baseline price prediction.

The pricing workflow is:

\[
\text{Base Price} =
\hat{f}(\text{Title, Category, Description, Units, Rating})
\]

For a fixed-dollar discount:

\[
\text{Final Price} =
\max(0.50,\ \text{Base Price}-\text{Discount})
\]

For a percentage discount \(d\):

\[
\text{Final Price} =
\max\left(0.50,\ \text{Base Price}\left(1-\frac{d}{100}\right)\right)
\]

These formulas assume a minimum displayed price of $0.50 and valid discount inputs.

---

## 📊 Step 5: Model Performance Benchmarks

The following table contains the reported evaluation results for a held-out test set of 176 products.

| Model | Variant | 5-Fold CV R² (log) | Test MAE | Test RMSE | Test MAPE | Test R² (log) |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest (Tuned)** | **B** | **0.8994** | **$23.00** | **$91.76** | **23.52%** | **0.8654** |
| Random Forest (Base) | B | 0.8443 | $23.12 | $93.75 | 23.73% | 0.8655 |
| Gradient Boosting | B | 0.8462 | $24.71 | $98.12 | 24.46% | 0.8535 |
| Ridge Regression | B | 0.8173 | $25.29 | $107.03 | 27.10% | 0.8248 |
| LightGBM | B | 0.8247 | $25.96 | $109.15 | 27.07% | 0.8186 |
| XGBoost | B | 0.8378 | $26.11 | $109.28 | 26.96% | 0.8183 |
| Random Forest | A | 0.8443 | $23.20 | $93.58 | 23.82% | 0.8661 |
| Gradient Boosting | A | 0.8508 | $24.58 | $98.80 | 24.30% | 0.8513 |

**Note:** Verify the reported cross-validation and test metrics against the training artifacts before presenting them as final results.

### Hyperparameter Tuning (Optuna)

- **Framework:** Optuna with a Tree-structured Parzen Estimator (TPE) sampler.
- **Reported Optimal Parameters:**
  - `n_estimators=160`
  - `max_depth=24`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `max_features=0.468`
  - `n_components=50`

### Top Reported Feature Drivers

1. Rating-missing indicators and review counts.
2. Latent text features such as `Text_Semantic_SVD_5` and `Text_Semantic_SVD_2`.
3. Product subcategories, including `Meat & Seafood`.
4. Text length and title length.
5. Product weight and item count.

---

## 💻 Step 6: Streamlit Web Application

The Streamlit interface, implemented in `app.py`, provides the following features.

### Interactive Product Input

- Product title.
- Category selection from 19 subcategories.
- Customer rating slider.
- Unrated-product checkbox.
- Product features and description.

### Quick-Fill Templates

- Olive Oil
- Salmon Case
- Prebiotic Soda
- Caviar

### Promotional Pricing

- Dollar-based discounts (`$`).
- Percentage-based discounts (`%`).
- Base price, applied discount, and final member price.

### Prediction Intervals

Displays empirical 80% and 90% prediction intervals to communicate uncertainty. These are not guaranteed price ranges.

### Category Benchmarking

An interactive Plotly boxplot compares the predicted price with the corresponding category distribution.

### Model Insights & EDA

- Model performance metrics.
- Feature importance visualizations.
- Exploratory data analysis charts.

---

## 🚀 Setup & Quickstart Guide

### 1. Prerequisites

- Python 3.10 or later.
- Git (optional).

Package compatibility depends on the versions specified in `requirements.txt`.

### 2. Environment Setup

**Windows PowerShell**

```powershell
cd "C:\path\to\store"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd /path/to/store

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py
```

The training script is intended to perform preprocessing, cross-validation, model comparison, hyperparameter tuning, and artifact export.

Expected artifacts:

```text
artifacts/
├── model.joblib
└── metadata.json
```

Actual outputs depend on the implementation of `src/train.py`.

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Run Standalone Inference

```bash
python src/predict.py
```

This command requires `src/predict.py` to define an executable entry point.

---

## ⚠️ Model Limitations & Real-World Considerations

### 1. Catalog Regionality

Costco prices may vary by warehouse location and region. Online prices may include delivery-related costs. The dataset reflects the pricing represented in the scraped online catalog.

### 2. Dataset Size

The dataset contains approximately 1,752 cleaned products. It may not represent the full range of products, locations, seasons, or promotions. Performance on unseen products should be evaluated before practical deployment.

### 3. Text Representation Scalability

TF-IDF with Truncated SVD provides a compact representation without requiring a large transformer model. Actual inference latency depends on hardware and implementation and should be benchmarked.

### 4. High-Value Products

Log transformation reduces the influence of high prices during training but does not eliminate prediction errors. Premium and bulk products can have larger absolute dollar errors than everyday groceries.

### 5. Discount Assumptions

Promotional prices may depend on membership eligibility, purchase quantities, coupon conditions, and promotion dates. The application does not independently verify current Costco offers.

### 6. Model Monitoring and Retraining

Catalog prices and product availability change over time. A real-world deployment should monitor prediction errors, detect data drift, validate new data, and retrain when necessary.

---

## 📄 License & Disclaimer

This project is intended for educational, analytical, and demonstration purposes. Predictions are estimates based on the available dataset and are not guaranteed to match current Costco prices.

Costco is a registered trademark of Costco Wholesale Corporation. This project is not affiliated with or endorsed by Costco Wholesale Corporation.
