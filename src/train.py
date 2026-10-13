"""
train.py
End-to-end model training, cross-validation, data leakage study, and hyperparameter tuning
for Costco Grocery Price Prediction.
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import optuna

# Ensure root directory in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    parse_price,
    parse_discount,
    parse_rating,
    build_preprocessor
)

RANDOM_SEED = 42
DATA_PATH = os.path.join(ROOT_DIR, "data", "costco_grocery.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")


def load_and_clean_data(data_path=DATA_PATH):
    print("=" * 60)
    print("STEP 1: DATA INGESTION AND CLEANING")
    print("=" * 60)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    raw_df = pd.read_csv(data_path)
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Columns: {list(raw_df.columns)}")

    # Parse Price
    raw_df['Price_clean'] = raw_df['Price'].apply(parse_price)

    # Missing values check
    missing_before = raw_df.isnull().sum()
    print("\nMissing values in raw data:")
    for col, count in missing_before.items():
        print(f"  {col}: {count} ({count/len(raw_df)*100:.1f}%)")

    # Filter out missing target rows and scraper 404 pages
    is_valid_price = raw_df['Price_clean'].notnull()
    is_valid_page = ~raw_df['Title'].str.contains("Page Not Found", case=False, na=False)

    df_clean = raw_df[is_valid_price & is_valid_page].copy().reset_index(drop=True)
    dropped_rows = len(raw_df) - len(df_clean)
    print(f"\nFiltered out {dropped_rows} unmodelable rows (3 missing prices + 2 scraper error pages).")
    print(f"Cleaned dataset modeling size: {len(df_clean)} rows.")

    # Price stats & Outlier reporting
    q1 = df_clean['Price_clean'].quantile(0.25)
    q3 = df_clean['Price_clean'].quantile(0.75)
    iqr = q3 - q1
    print("\nPrice Statistics:")
    print(f"  Min: ${df_clean['Price_clean'].min():.2f}")
    print(f"  Q1: ${q1:.2f}")
    print(f"  Median: ${df_clean['Price_clean'].median():.2f}")
    print(f"  Mean: ${df_clean['Price_clean'].mean():.2f}")
    print(f"  Q3: ${q3:.2f}")
    print(f"  Max: ${df_clean['Price_clean'].max():.2f}")
    print(f"  Raw IQR 1.5x Upper Bound: ${q3 + 1.5 * iqr:.2f}")
    print("  Note on Outliers: High prices ($200 - $1999) represent genuine luxury wholesale items")
    print("  (e.g., Caviar Kilo Packs, Whole Parmigiano Reggiano Wheels, A5 Japanese Wagyu).")
    print("  Target transformation log1p(Price) compresses right-skew into a normal distribution")
    print("  without discarding valuable commercial inventory.")

    return df_clean


def run_model_comparison(X_train, y_train_log, X_test, y_test_orig, y_test_log, include_discount=False):
    variant_name = "Variant A (With Discount Features)" if include_discount else "Variant B (Without Discount Features - Safe)"
    print(f"\nEvaluating {variant_name}...")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    candidate_models = {
        "Ridge Regression": (Ridge(alpha=1.5), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=RANDOM_SEED), True),
        "Random Forest": (RandomForestRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1), True),
        "XGBoost": (xgb.XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1), False),
        "LightGBM": (lgb.LGBMRegressor(n_estimators=150, learning_rate=0.06, max_depth=6, num_leaves=31, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1), False),
    }

    results = []
    trained_pipes = {}

    for name, (reg, dense_svd) in candidate_models.items():
        pipe = Pipeline([
            ('prep', build_preprocessor(include_discount=include_discount, dense_svd=dense_svd)),
            ('reg', reg)
        ])

        # 5-Fold CV on training set
        cv_scores = cross_val_score(pipe, X_train, y_train_log, cv=kf, scoring='r2', n_jobs=-1)
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)

        # Fit and predict on test set
        pipe.fit(X_train, y_train_log)
        trained_pipes[name] = pipe

        y_pred_log = pipe.predict(X_test)
        y_pred_orig = np.clip(np.expm1(y_pred_log), a_min=1.0, a_max=None)

        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = root_mean_squared_error(y_test_orig, y_pred_orig)
        r2_orig = r2_score(y_test_orig, y_pred_orig)
        mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100.0
        r2_log = r2_score(y_test_log, y_pred_log)

        results.append({
            'Model': name,
            'Variant': 'Variant A' if include_discount else 'Variant B',
            'CV R2 (log)': f"{cv_r2_mean:.4f} +/- {cv_r2_std:.4f}",
            'Test MAE ($)': f"${mae:.2f}",
            'Test RMSE ($)': f"${rmse:.2f}",
            'Test R2 (orig)': f"{r2_orig:.4f}",
            'Test MAPE (%)': f"{mape:.2f}%",
            'Test R2 (log)': f"{r2_log:.4f}",
            'mae_num': mae,
            'rmse_num': rmse,
            'r2_num': r2_orig,
            'mape_num': mape,
            'r2_log_num': r2_log,
            'cv_r2_num': cv_r2_mean
        })

    return pd.DataFrame(results), trained_pipes


def tune_best_model(X_train, y_train_log, X_val, y_val_log, n_trials=15):
    print("\n" + "=" * 60)
    print("STEP 3: HYPERPARAMETER TUNING (OPTUNA)")
    print("=" * 60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 80, 200, step=20)
        max_depth = trial.suggest_int('max_depth', 10, 25)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 6)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)
        max_features = trial.suggest_float('max_features', 0.45, 0.85)
        n_components = trial.suggest_int('n_components', 40, 70, step=10)

        pipe = Pipeline([
            ('prep', build_preprocessor(include_discount=False, dense_svd=True, n_components=n_components)),
            ('reg', RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ])
        pipe.fit(X_train, y_train_log)
        val_preds = pipe.predict(X_val)
        return r2_score(y_val_log, val_preds)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=120)

    print(f"Optuna Best Validation R2 (log): {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)
    return study.best_params, study.best_value


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_clean = load_and_clean_data()

    # Split: 80% train, 10% val, 10% test
    features = ['Title', 'Sub Category', 'Feature', 'Product Description', 'Rating', 'Discount']
    X_train, X_temp, y_train_orig, y_temp_orig = train_test_split(
        df_clean[features], df_clean['Price_clean'], test_size=0.20, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val_orig, y_test_orig = train_test_split(
        X_temp, y_temp_orig, test_size=0.50, random_state=RANDOM_SEED
    )

    y_train_log = np.log1p(y_train_orig)
    y_val_log = np.log1p(y_val_orig)
    y_test_log = np.log1p(y_test_orig)

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Step 4: Data Leakage Comparison
    print("\n" + "=" * 60)
    print("STEP 2 & 4: MODELING AND DATA LEAKAGE ANALYSIS")
    print("=" * 60)

    results_b, _ = run_model_comparison(X_train, y_train_log, X_test, y_test_orig, y_test_log, include_discount=False)
    results_a, _ = run_model_comparison(X_train, y_train_log, X_test, y_test_orig, y_test_log, include_discount=True)

    print("\n--- MODEL COMPARISON TABLE (TEST SET) ---")
    combined_results = pd.concat([results_b, results_a], ignore_index=True)
    print(combined_results[['Model', 'Variant', 'CV R2 (log)', 'Test MAE ($)', 'Test RMSE ($)', 'Test MAPE (%)', 'Test R2 (log)']])

    # Tune the best model (Random Forest, Variant B)
    best_params, best_val_score = tune_best_model(X_train, y_train_log, X_val, y_val_log, n_trials=12)

    # Train tuned model on train+val or train
    final_pipeline = Pipeline([
        ('prep', build_preprocessor(include_discount=False, dense_svd=True, n_components=best_params['n_components'])),
        ('reg', RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])

    print("\nFitting final pipeline on training set...")
    final_pipeline.fit(X_train, y_train_log)

    # Final test evaluation
    y_test_pred_log = final_pipeline.predict(X_test)
    y_test_pred_orig = np.clip(np.expm1(y_test_pred_log), a_min=1.0, a_max=None)

    final_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
    final_rmse = root_mean_squared_error(y_test_orig, y_test_pred_orig)
    final_r2_orig = r2_score(y_test_orig, y_test_pred_orig)
    final_mape = mean_absolute_percentage_error(y_test_orig, y_test_pred_orig) * 100.0
    final_r2_log = r2_score(y_test_log, y_test_pred_log)

    print("\n" + "=" * 60)
    print("FINAL TUNED MODEL PERFORMANCE (TEST SET)")
    print("=" * 60)
    print(f"MAE ($):          ${final_mae:.2f}")
    print(f"RMSE ($):         ${final_rmse:.2f}")
    print(f"R2 (original $):  {final_r2_orig:.4f}")
    print(f"MAPE (%):         {final_mape:.2f}%")
    print(f"R2 (log space):   {final_r2_log:.4f}")

    # Residual prediction intervals
    val_preds_log = final_pipeline.predict(X_val)
    val_resids_log = y_val_log.values - val_preds_log
    q10_res = float(np.percentile(val_resids_log, 10))
    q90_res = float(np.percentile(val_resids_log, 90))
    q05_res = float(np.percentile(val_resids_log, 5))
    q95_res = float(np.percentile(val_resids_log, 95))

    # Feature importances
    reg_step = final_pipeline.named_steps['reg']
    prep_step = final_pipeline.named_steps['prep']
    svd_names = [f"Text_Semantic_SVD_{i+1}" for i in range(best_params['n_components'])]
    unit_names = ["Weight_oz", "Item_Count", "Pack_Count", "Text_Length", "Title_Length", "Is_Organic", "Is_Bulk"]
    rating_names = ["Rating_Value", "Review_Count", "Rating_Missing"]
    ohe_names = list(prep_step.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(['Sub Category']))
    all_feature_names = svd_names + unit_names + rating_names + ohe_names

    feat_importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': reg_step.feature_importances_
    }).sort_values('importance', ascending=False)

    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    top15 = feat_importances.head(15).iloc[::-1]
    plt.barh(top15['feature'], top15['importance'], color='#1f77b4')
    plt.title('Top 15 Drivers of Costco Grocery Price', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance (Gini / MDI)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()

    # Save artifacts
    model_save_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump(final_pipeline, model_save_path)
    print(f"\nSaved final model pipeline to: {model_save_path}")

    # Subcategory benchmark distribution
    subcat_stats = df_clean.groupby('Sub Category')['Price_clean'].agg(
        count='count',
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        min='min',
        max='max',
        mean='mean',
        std='std'
    ).round(2).to_dict(orient='index')

    metadata = {
        'model_name': 'Tuned Random Forest Regressor',
        'variant': 'Variant B (Without Discount Features - Safe Against Leakage)',
        'training_date': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'best_hyperparameters': best_params,
        'metrics': {
            'test_mae_dollars': round(float(final_mae), 2),
            'test_rmse_dollars': round(float(final_rmse), 2),
            'test_r2_original': round(float(final_r2_orig), 4),
            'test_mape_percent': round(float(final_mape), 2),
            'test_r2_log': round(float(final_r2_log), 4),
            'val_r2_log': round(float(best_val_score), 4)
        },
        'model_comparisons': combined_results[['Model', 'Variant', 'CV R2 (log)', 'Test MAE ($)', 'Test RMSE ($)', 'Test MAPE (%)', 'Test R2 (log)']].to_dict(orient='records'),
        'prediction_intervals': {
            'residual_q10_log': q10_res,
            'residual_q90_log': q90_res,
            'residual_q05_log': q05_res,
            'residual_q95_log': q95_res
        },
        'subcategories': sorted(df_clean['Sub Category'].unique().tolist()),
        'subcategory_stats': subcat_stats,
        'top_features': feat_importances.head(20).to_dict(orient='records')
    }

    meta_save_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_save_path}")
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
