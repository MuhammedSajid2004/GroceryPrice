"""
preprocessing.py
Custom scikit-learn transformers and preprocessing pipeline for Costco Grocery dataset.
Ensures identical transformations between training and live inference.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def parse_price(val):
    """
    Parses raw price strings (e.g. '$56.99', '$32.99through-$83.99', '1,200.50') into a float.
    For price ranges, computes the midpoint.
    """
    if pd.isnull(val):
        return np.nan
    s = str(val).strip().replace(',', '')
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
    if len(nums) == 0:
        return np.nan
    elif len(nums) == 1:
        return nums[0]
    else:
        # Midpoint of price range
        return sum(nums[:2]) / 2.0


def parse_discount(val):
    """
    Parses discount strings into (discount_amount, discount_percent, has_discount).
    Handles patterns such as 'After $5 OFF', 'After $40 - $80 OFF', '15%', 'No Discount'.
    """
    if pd.isnull(val):
        return 0.0, 0.0, 0
    s = str(val).strip().replace(',', '')
    if s in ['No Discount', '.', 'This item is not returnable.', 'Limit 1 Per Member', 'Limit 5 Per Member', '']:
        return 0.0, 0.0, 0
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
    if 'OFF' in s.upper() and len(nums) > 0:
        amount = sum(nums[:2]) / 2.0 if len(nums) >= 2 and '-' in s else nums[0]
        return float(amount), 0.0, 1
    if '%' in s and len(nums) > 0:
        return 0.0, float(nums[0]), 1
    return 0.0, 0.0, 0


def parse_rating(val):
    """
    Parses customer rating strings into (rating_value, review_count).
    Example: 'Rated 4.3 out of 5 stars based on 265 reviews.' -> (4.3, 265)
    """
    if pd.isnull(val):
        return np.nan, 0
    s = str(val).strip()
    match = re.search(r"Rated\s+([0-9.]+)\s+out of 5 stars based on\s+([0-9]+)\s+reviews", s)
    if match:
        return float(match.group(1)), int(match.group(2))
    return np.nan, 0


def get_combined_text(df):
    """
    Combines Title, Feature, and Product Description into a single corpus string.
    Works for both DataFrames and Series.
    """
    if isinstance(df, pd.DataFrame):
        title = df['Title'].fillna('').astype(str) if 'Title' in df.columns else pd.Series('', index=df.index)
        feature = df['Feature'].fillna('').astype(str) if 'Feature' in df.columns else pd.Series('', index=df.index)
        desc = df['Product Description'].fillna('').astype(str) if 'Product Description' in df.columns else pd.Series('', index=df.index)
        return title + ' ' + feature + ' ' + desc
    elif isinstance(df, pd.Series):
        return df.fillna('').astype(str)
    return pd.Series(df).fillna('').astype(str)


class UnitFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts physical pack size, weight, item count, and bulk indicators from product text.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = get_combined_text(X)
        elif isinstance(X, pd.Series):
            texts = X
        else:
            texts = pd.Series(X.ravel() if hasattr(X, 'ravel') else X)

        features = []
        for text in texts:
            t = str(text).lower()

            # 1. Weight in ounces (normalized from lbs, kg, grams, fl oz)
            weight_oz = np.nan
            lb_m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\b', t)
            if lb_m:
                weight_oz = float(lb_m[0]) * 16.0
            else:
                oz_m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:fl\.?\s*oz\.?|ounces?|oz\.?)\b', t)
                if oz_m:
                    weight_oz = float(oz_m[0])
                else:
                    kg_m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b', t)
                    if kg_m:
                        weight_oz = float(kg_m[0]) * 35.274
                    else:
                        g_m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:g|grams?)\b', t)
                        if g_m:
                            weight_oz = float(g_m[0]) * 0.035274

            # 2. Item count / servings / loads / pieces
            item_count = np.nan
            cnt_m = re.findall(r'(\d+)\s*(?:-\s*)?(?:count|ct|servings?|loads?|pieces?|slices?|capsules?|tablets?|bars?|tins?)\b', t)
            if cnt_m:
                item_count = float(cnt_m[0])

            # 3. Pack count (e.g. 2-pack, 6 pack, 12-pack)
            pack_count = np.nan
            pk_m = re.findall(r'(\d+)\s*(?:-\s*)?(?:pack|pk|case)\b', t)
            if pk_m:
                pack_count = float(pk_m[0])

            # 4. Text lengths
            text_len = len(t)
            first_line = str(text).split('\n')[0]
            title_len = len(first_line)

            # 5. Key categorical keywords
            has_organic = 1.0 if 'organic' in t else 0.0
            has_bulk = 1.0 if any(w in t for w in ['bulk', 'case', 'total', 'lbs']) else 0.0

            features.append([
                weight_oz,
                item_count,
                pack_count,
                text_len,
                title_len,
                has_organic,
                has_bulk
            ])
        return np.array(features)


class RatingExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numerical rating, review count, and missing rating flag.
    Handles both raw scraped string format and numeric inputs from user interfaces.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = X if isinstance(X, pd.Series) else pd.Series(X.ravel() if hasattr(X, 'ravel') else X)
        out = []
        for val in series:
            if pd.isnull(val):
                out.append([np.nan, 0.0, 1.0])
            elif isinstance(val, (int, float)):
                out.append([float(val), 50.0, 0.0])
            else:
                s = str(val).strip()
                m = re.search(r"Rated\s+([0-9.]+)\s+out of 5 stars based on\s+([0-9]+)\s+reviews", s)
                if m:
                    out.append([float(m.group(1)), float(m.group(2)), 0.0])
                else:
                    out.append([np.nan, 0.0, 1.0])
        return np.array(out)


class DiscountExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts discount amount, discount percent, and has_discount flag.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = X if isinstance(X, pd.Series) else pd.Series(X.ravel() if hasattr(X, 'ravel') else X)
        out = []
        for val in series:
            if pd.isnull(val):
                out.append([0.0, 0.0, 0.0])
            elif isinstance(val, (int, float)):
                out.append([float(val), 0.0, 1.0 if val > 0 else 0.0])
            else:
                s = str(val).strip().replace(',', '')
                if s in ['No Discount', '.', 'This item is not returnable.', 'Limit 1 Per Member', 'Limit 5 Per Member', '']:
                    out.append([0.0, 0.0, 0.0])
                else:
                    nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
                    if 'OFF' in s.upper() and len(nums) > 0:
                        amt = sum(nums[:2]) / 2.0 if len(nums) >= 2 and '-' in s else nums[0]
                        out.append([float(amt), 0.0, 1.0])
                    elif '%' in s and len(nums) > 0:
                        out.append([0.0, float(nums[0]), 1.0])
                    else:
                        out.append([0.0, 0.0, 0.0])
        return np.array(out)


def build_preprocessor(include_discount=False, dense_svd=True, n_components=50, max_features=4000):
    """
    Constructs a ColumnTransformer combining text TF-IDF + SVD, unit features, ratings, and subcategories.
    """
    # 1. Text pipeline (Title + Feature + Product Description)
    text_steps = [
        ('combine', FunctionTransformer(get_combined_text, validate=False)),
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, stop_words='english'))
    ]
    if dense_svd:
        text_steps.append(('svd', TruncatedSVD(n_components=n_components, random_state=42)))

    text_pipe = Pipeline(text_steps)

    # 2. Units & Quantity pipeline
    unit_pipe = Pipeline([
        ('combine', FunctionTransformer(get_combined_text, validate=False)),
        ('units', UnitFeatureExtractor()),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])

    # 3. Rating pipeline
    rating_pipe = Pipeline([
        ('rating_extractor', RatingExtractor()),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])

    # 4. Categorical Sub Category pipeline
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=not dense_svd))
    ])

    transformers = [
        ('text', text_pipe, ['Title', 'Feature', 'Product Description']),
        ('units', unit_pipe, ['Title', 'Feature', 'Product Description']),
        ('rating', rating_pipe, 'Rating'),
        ('cat', cat_pipe, ['Sub Category'])
    ]

    # Optional discount branch (for Variant A leakage study)
    if include_discount:
        disc_pipe = Pipeline([
            ('disc_extractor', DiscountExtractor()),
            ('scale', StandardScaler())
        ])
        transformers.append(('discount', disc_pipe, 'Discount'))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.3 if not dense_svd else 0.0
    )
    return preprocessor
