"""
train_eda_notebook.py
Standalone, fully-commented script version of the exploration and training notebook.
Can be executed directly or run cell-by-cell in VS Code / interactive editors (# %%).
"""

# %% [markdown]
# # Costco Grocery Price Prediction: End-to-End Walkthrough
# This script executes the complete data cleaning, feature engineering,
# modeling comparisons, leakage validation, and prediction pipeline.

# %%
import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import parse_price, parse_discount, parse_rating, build_preprocessor
from src.predict import predict_price

sns.set_theme(style='whitegrid')
print("Initialized pipeline environment.")

# %% [markdown]
# ## 1. Data Loading and Cleaning
df_raw = pd.read_csv(os.path.join(ROOT_DIR, "data", "costco_grocery.csv"))
print(f"Raw shape: {df_raw.shape}")

# Parse price and clean bad scraper entries
df_raw['Price_clean'] = df_raw['Price'].apply(parse_price)
df_clean = df_raw[df_raw['Price_clean'].notnull() & (~df_raw['Title'].str.contains('Page Not Found', case=False, na=False))].copy()
print(f"Cleaned shape: {df_clean.shape}")
print(df_clean['Price_clean'].describe())

# %% [markdown]
# ## 2. Model Comparisons & Leakage Check
# We load the recorded metrics from the 5-fold cross-validation runs
meta_path = os.path.join(ROOT_DIR, "artifacts", "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    comp_df = pd.DataFrame(meta['model_comparisons'])
    print("\n--- Model Benchmark Table ---")
    print(comp_df.to_string(index=False))
else:
    print("Metadata not found. Run `python src/train.py` first.")

# %% [markdown]
# ## 3. Production Inference Demonstration
sample_pred = predict_price(
    title="Kirkland Signature Wild Alaskan Sockeye Salmon Fillets, 3 lbs",
    sub_category="Meat & Seafood",
    feature="Individually vacuum packed, boneless, skin-on",
    product_description="Responsibly wild caught in Alaska, rich in Omega-3 fatty acids",
    rating=4.7,
    discount_val=5.0,
    discount_type="amount"
)

print("\n--- Live Test Prediction Output ---")
print(json.dumps(sample_pred, indent=2))
