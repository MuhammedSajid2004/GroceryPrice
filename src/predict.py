"""
predict.py
Inference module for predicting base and discounted Costco grocery prices.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import custom transformers so joblib unpickles correctly
import src.preprocessing as prep
from src.preprocessing import (
    parse_price,
    parse_discount,
    parse_rating,
    get_combined_text,
    UnitFeatureExtractor,
    RatingExtractor,
    DiscountExtractor
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "model.joblib")
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "metadata.json")

_model = None
_metadata = None


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """
    Loads and caches the model pipeline and metadata JSON.
    """
    global _model, _metadata
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        _model = joblib.load(model_path)

    if _metadata is None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

    return _model, _metadata


def predict_price(
    title: str,
    sub_category: str,
    feature: str = "",
    product_description: str = "",
    rating: float = None,
    discount_val: float = 0.0,
    discount_type: str = "amount",  # "amount" ($) or "percent" (%)
    model_path: str = MODEL_PATH,
    metadata_path: str = METADATA_PATH
) -> dict:
    """
    End-to-end price prediction function.
    Returns predicted base price, final discounted price, prediction intervals,
    and category benchmark comparison.
    """
    model, metadata = load_artifacts(model_path, metadata_path)

    # 1. Validation & Input Sanitization
    if not title or str(title).strip() == "":
        raise ValueError("Product Title cannot be empty.")
    if not sub_category or str(sub_category).strip() == "":
        raise ValueError("Sub Category must be selected.")

    # Prepare single-row DataFrame for pipeline
    input_row = pd.DataFrame([{
        'Title': str(title).strip(),
        'Sub Category': str(sub_category).strip(),
        'Feature': str(feature).strip() if feature else "",
        'Product Description': str(product_description).strip() if product_description else "",
        'Rating': rating if rating is not None else np.nan,
        'Discount': 'No Discount'
    }])

    # 2. Predict Base Price on log scale and invert
    log_pred = float(model.predict(input_row)[0])
    base_price = float(np.expm1(log_pred))
    base_price = max(1.0, round(base_price, 2))

    # 3. Calculate Discount and Final Price
    discount_amount = 0.0
    discount_percent = 0.0

    if discount_val is not None and discount_val > 0:
        if discount_type == "percent":
            discount_percent = float(min(100.0, discount_val))
            discount_amount = round(base_price * (discount_percent / 100.0), 2)
        else:
            discount_amount = float(min(base_price, discount_val))
            discount_percent = round((discount_amount / base_price) * 100.0, 1) if base_price > 0 else 0.0

    final_price = max(0.50, round(base_price - discount_amount, 2))

    # 4. Compute Prediction Intervals using Residual Percentiles
    intervals = metadata.get('prediction_intervals', {
        'residual_q10_log': -0.3631,
        'residual_q90_log': 0.3522,
        'residual_q05_log': -0.5509,
        'residual_q95_log': 0.4807
    })

    # 80% interval
    lower_80 = max(1.0, round(float(np.expm1(log_pred + intervals['residual_q10_log'])), 2))
    upper_80 = max(lower_80, round(float(np.expm1(log_pred + intervals['residual_q90_log'])), 2))

    # 90% interval
    lower_90 = max(1.0, round(float(np.expm1(log_pred + intervals['residual_q05_log'])), 2))
    upper_90 = max(lower_90, round(float(np.expm1(log_pred + intervals['residual_q95_log'])), 2))

    # 5. Category Price Benchmark
    subcat_stats = metadata.get('subcategory_stats', {}).get(sub_category, None)
    comparison = {}
    if subcat_stats:
        cat_median = subcat_stats['median']
        cat_q25 = subcat_stats['q25']
        cat_q75 = subcat_stats['q75']
        cat_min = subcat_stats['min']
        cat_max = subcat_stats['max']

        diff_from_median = round(base_price - cat_median, 2)
        pct_diff = round(((base_price - cat_median) / cat_median) * 100.0, 1)

        if base_price < cat_q25:
            position = "Below Average (Budget-Friendly for this category)"
        elif base_price <= cat_q75:
            position = "Within Typical Price Range (Mid-tier)"
        else:
            position = "Above Typical Range (Premium/Bulk offering)"

        comparison = {
            'category': sub_category,
            'cat_median': cat_median,
            'cat_q25': cat_q25,
            'cat_q75': cat_q75,
            'cat_min': cat_min,
            'cat_max': cat_max,
            'diff_from_median': diff_from_median,
            'pct_diff_from_median': pct_diff,
            'position': position
        }

    return {
        'base_price': base_price,
        'discount_amount': discount_amount,
        'discount_percent': discount_percent,
        'final_price': final_price,
        'interval_80': (lower_80, upper_80),
        'interval_90': (lower_90, upper_90),
        'category_comparison': comparison
    }


if __name__ == '__main__':
    # Quick sanity test
    test_result = predict_price(
        title="Kirkland Signature Organic Extra Virgin Olive Oil, 2 L",
        sub_category="Pantry & Dry Goods",
        feature="Cold extracted, 100% Italian olives",
        product_description="Rich taste, kosher, perfect for cooking and salads",
        rating=4.8,
        discount_val=3.0,
        discount_type="amount"
    )
    print("Sanity Test Prediction Result:")
    print(json.dumps(test_result, indent=2))
