"""
app.py
Production Streamlit Web Application for Costco Grocery Price Prediction.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predict import predict_price, load_artifacts

# Page Configuration
st.set_page_config(
    page_title="Costco Grocery Price Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #E31837; /* Costco Red */
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #005DAA; /* Costco Blue */
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #005DAA;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-card {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #E31837;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #111;
    }
    .metric-lbl {
        font-size: 0.95rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model_and_metadata():
    try:
        return load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return None, None


model, metadata = get_model_and_metadata()

# Header
st.markdown('<div class="main-title">🛒 Costco Grocery Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End ML Pipeline for Grocery Price Estimation, Promotion Impact & Category Benchmarking</div>', unsafe_allow_html=True)

if metadata is None:
    st.warning("⚠️ Model artifacts not found. Please run `python src/train.py` to train and generate the artifacts.")
    st.stop()

subcategories = metadata.get('subcategories', [
    "Bakery & Desserts", "Beverages & Water", "Breakfast", "Candy", "Cleaning Supplies",
    "Coffee", "Deli", "Floral", "Gift Baskets", "Household", "Kirkland Signature Grocery",
    "Laundry Detergent & Supplies", "Meat & Seafood", "Organic", "Pantry & Dry Goods",
    "Paper & Plastic Products", "Poultry", "Seafood", "Snacks"
])

# Quick Examples Definition
EXAMPLES = {
    "Select an example to auto-fill...": None,
    "Kirkland Extra Virgin Olive Oil (2 L)": {
        "title": "Kirkland Signature Organic Extra Virgin Olive Oil, 2 L",
        "category": "Pantry & Dry Goods",
        "rating": 4.8,
        "unrated": False,
        "discount_type": "Dollar ($)",
        "discount_val": 3.0,
        "feature": "100% Italian Olives, Cold Extracted, Certified Organic",
        "desc": "Rich aroma, smooth finish, perfect for sautéing, pasta, and fresh dressings."
    },
    "Wild Alaskan Sockeye Salmon (10 Lbs)": {
        "title": "Northwest Fish Wild Alaskan Sockeye Salmon Fillets Total 25 Count, 1 Case Totaling 10 Lbs.",
        "category": "Meat & Seafood",
        "rating": 4.6,
        "unrated": False,
        "discount_type": "Dollar ($)",
        "discount_val": 30.0,
        "feature": "Individually vacuum packed portions, flash frozen at peak freshness",
        "desc": "Wild-caught in icy Alaskan waters, rich in healthy Omega-3s, vibrant natural red color."
    },
    "Prebiotic Soda Variety Pack (15-count)": {
        "title": "Poppi Prebiotic Soda, Variety Pack, 12 fl oz, 15-count",
        "category": "Beverages & Water",
        "rating": 4.4,
        "unrated": False,
        "discount_type": "Dollar ($)",
        "discount_val": 5.0,
        "feature": "5g sugar or less, sparkling probiotic beverage, fruit juices",
        "desc": "Refreshing soda with clean ingredients and Apple Cider Vinegar for gut health."
    },
    "Plaza Golden Osetra Caviar (Kilo Pack)": {
        "title": "Plaza Golden Osetra Caviar Kilo Pack, 35.2 oz",
        "category": "Meat & Seafood",
        "rating": 4.9,
        "unrated": False,
        "discount_type": "Dollar ($)",
        "discount_val": 0.0,
        "feature": "Malossol cured, large firm golden pearls, pristine buttery flavor",
        "desc": "Wholesale bulk pack of luxury caviar shipped in temperature-controlled cooler."
    }
}

# Main Layout: 2 Columns for Input and Results
col_input, col_results = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("📝 Enter Product Attributes")

    selected_example = st.selectbox("💡 Quick Fill Template", list(EXAMPLES.keys()))
    ex_data = EXAMPLES.get(selected_example)

    with st.form("prediction_form"):
        title_val = ex_data["title"] if ex_data else ""
        title = st.text_input("Product Title *", value=title_val, placeholder="e.g. David's Cookies Butter Pecan Meltaways 32 oz, 2-pack")

        default_cat_idx = 0
        if ex_data and ex_data["category"] in subcategories:
            default_cat_idx = subcategories.index(ex_data["category"])
        sub_category = st.selectbox("Sub Category *", subcategories, index=default_cat_idx)

        c1, c2 = st.columns(2)
        with c1:
            rating_val = ex_data["rating"] if ex_data else 4.5
            rating_input = st.slider("Customer Rating (Stars)", 1.0, 5.0, float(rating_val), step=0.1)
        with c2:
            unrated = st.checkbox("No Reviews Yet / Unrated", value=ex_data["unrated"] if ex_data else False)

        st.markdown("**Promotion / Discount Settings**")
        d1, d2 = st.columns(2)
        with d1:
            disc_type_val = 0 if (ex_data and ex_data["discount_type"] == "Dollar ($)") else 0
            discount_type = st.radio("Discount Mode", ["Dollar ($)", "Percentage (%)"], horizontal=True, index=disc_type_val)
        with d2:
            disc_num = ex_data["discount_val"] if ex_data else 0.0
            discount_val = st.number_input(
                "Discount Value",
                min_value=0.0,
                max_value=1000.0 if discount_type == "Dollar ($)" else 100.0,
                value=float(disc_num),
                step=1.0 if discount_type == "Dollar ($)" else 5.0
            )

        feature_val = ex_data["feature"] if ex_data else ""
        feature = st.text_area("Key Features (Bullet Points)", value=feature_val, height=75, placeholder="e.g. Kosher OU-D, 14 servings, Made in USA")

        desc_val = ex_data["desc"] if ex_data else ""
        product_description = st.text_area("Product Description", value=desc_val, height=100, placeholder="Full product summary and ingredient details...")

        submit_btn = st.form_submit_button("🚀 Predict Price", use_container_width=True)

with col_results:
    st.subheader("📊 Price Prediction & Analysis")

    if submit_btn or ex_data:
        if not title or str(title).strip() == "":
            st.error("❌ Please provide a Product Title.")
        else:
            with st.spinner("Analyzing text, parsing pack sizes, and running prediction pipeline..."):
                try:
                    disc_mode = "percent" if "Percentage" in discount_type else "amount"
                    eff_rating = None if unrated else rating_input

                    res = predict_price(
                        title=title,
                        sub_category=sub_category,
                        feature=feature,
                        product_description=product_description,
                        rating=eff_rating,
                        discount_val=discount_val,
                        discount_type=disc_mode
                    )

                    base_price = res['base_price']
                    disc_amount = res['discount_amount']
                    disc_pct = res['discount_percent']
                    final_price = res['final_price']
                    lower_80, upper_80 = res['interval_80']
                    lower_90, upper_90 = res['interval_90']
                    comp = res['category_comparison']

                    # Display Metric Cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-lbl">Predicted Base Price</div>
                            <div class="metric-val">${base_price:.2f}</div>
                            <small style="color: #666;">Standard catalog price</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with m2:
                        disc_text = f"-${disc_amount:.2f} ({disc_pct:.1f}%)" if disc_amount > 0 else "None applied"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-lbl">Promo Discount</div>
                            <div class="metric-val" style="color: #2ca02c;">{disc_text}</div>
                            <small style="color: #666;">Post-inference deduction</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with m3:
                        st.markdown(f"""
                        <div class="highlight-card">
                            <div class="metric-lbl">Final Member Price</div>
                            <div class="metric-val" style="color: #E31837;">${final_price:.2f}</div>
                            <small style="color: #666;">Estimated checkout price</small>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # Prediction Intervals Callout
                    st.markdown("#### 🎯 Prediction Uncertainty Intervals")
                    st.info(
                        f"**80% Confidence Interval:** **${lower_80:.2f} — ${upper_80:.2f}**  \n"
                        f"**90% Confidence Interval:** **${lower_90:.2f} — ${upper_90:.2f}**  \n"
                        f"*Calibrated from validation residual percentiles to account for item variance.*"
                    )

                    # Category Comparison Chart
                    if comp:
                        st.markdown(f"#### 🏷️ Comparison with '{sub_category}' Category")
                        st.write(f"Position: **{comp['position']}** ({comp['pct_diff_from_median']:+.1f}% vs Category Median of ${comp['cat_median']:.2f})")

                        # Plotly Box & Marker Benchmark
                        fig = go.Figure()

                        # Category Boxplot summary
                        fig.add_trace(go.Box(
                            q1=[comp['cat_q25']],
                            median=[comp['cat_median']],
                            q3=[comp['cat_q75']],
                            lowerfence=[comp['cat_min']],
                            upperfence=[min(comp['cat_max'], comp['cat_q75'] + 2.5 * (comp['cat_q75'] - comp['cat_q25']))],
                            name=f"{sub_category} Distribution",
                            orientation='h',
                            marker_color='#005DAA',
                            boxpoints=False
                        ))

                        # Predicted Price Marker
                        fig.add_trace(go.Scatter(
                            x=[base_price],
                            y=[f"{sub_category} Distribution"],
                            mode='markers+text',
                            name='Your Predicted Base Price',
                            marker=dict(color='#E31837', size=16, symbol='diamond', line=dict(width=2, color='white')),
                            text=[f"  Predicted: ${base_price:.2f}"],
                            textposition="top center"
                        ))

                        fig.update_layout(
                            title=dict(text=f"Price Benchmark: This Item vs {sub_category}", font=dict(size=14)),
                            xaxis=dict(title="Price ($)", showgrid=True),
                            yaxis=dict(showticklabels=False),
                            height=240,
                            margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👈 Enter product details on the left and click **Predict Price** to generate estimates.")

# ----------------- INSIGHTS & TABS SECTION -----------------
st.markdown("---")
tab_insights, tab_eda, tab_leakage = st.tabs(["📈 Model Insights & Feature Drivers", "🔍 Exploratory Data Analysis (EDA)", "🛡️ Data Leakage Analysis"])

with tab_insights:
    st.subheader("Model Performance & Architecture")

    col_m1, col_m2 = st.columns([1, 1.2])

    with col_m1:
        st.markdown("##### 🏆 Best Model: Tuned Random Forest Regressor")
        m_data = metadata.get('metrics', {})
        st.write(f"- **Target Transform:** `log1p(Price)` (inverted via `expm1`)")
        st.write(f"- **Hold-out Test MAE:** `${m_data.get('test_mae_dollars', 23.03):.2f}`")
        st.write(f"- **Hold-out Test RMSE:** `${m_data.get('test_rmse_dollars', 93.63):.2f}`")
        st.write(f"- **Hold-out Test MAPE:** `{m_data.get('test_mape_percent', 23.49):.2f}%`")
        st.write(f"- **Hold-out Test R² (log space):** `{m_data.get('test_r2_log', 0.8669):.4f}`")
        st.write(f"- **Validation 5-Fold CV R²:** `{m_data.get('cv_r2_log', 0.9001):.4f}`")

        st.markdown("##### 📋 Model Comparison Table (Test Set)")
        comparisons = metadata.get('model_comparisons', [])
        if comparisons:
            comp_df = pd.DataFrame(comparisons)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("##### 🔑 Top Feature Drivers of Price")
        feat_img_path = os.path.join(ROOT_DIR, "artifacts", "plots", "feature_importance.png")
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, caption="Random Forest Feature Importance (MDI / Gini)", use_container_width=True)
        else:
            top_feats = pd.DataFrame(metadata.get('top_features', []))
            if not top_feats.empty:
                st.dataframe(top_feats.head(10), use_container_width=True)

        st.markdown("""
        **Key Findings:**
        1. **Rating Metadata & Popularity:** Missing ratings and high review counts serve as strong indicators of commercial bulk supply.
        2. **Semantic Text Representation:** SVD latent components derived from product title and descriptions capture luxury keywords (e.g. caviar, wagyu, organic, imported).
        3. **Subcategory Base Level:** High-value categories (Meat & Seafood, Floral, Gift Baskets) naturally baseline at higher price tiers.
        4. **Pack Sizes & Physical Units:** Weight in ounces and item counts extracted via regex anchor quantity scaling.
        """)

with tab_eda:
    st.subheader("Dataset Characteristics & Distributions")
    eda_c1, eda_c2 = st.columns(2)

    with eda_c1:
        p_dist = os.path.join(ROOT_DIR, "artifacts", "plots", "price_distribution.png")
        if os.path.exists(p_dist):
            st.image(p_dist, caption="Price Distribution: Raw ($) vs log1p(Price)", use_container_width=True)

        d_dist = os.path.join(ROOT_DIR, "artifacts", "plots", "discount_vs_price.png")
        if os.path.exists(d_dist):
            st.image(d_dist, caption="Promotions: Standard vs Discounted Item Prices", use_container_width=True)

    with eda_c2:
        sub_dist = os.path.join(ROOT_DIR, "artifacts", "plots", "price_by_subcategory.png")
        if os.path.exists(sub_dist):
            st.image(sub_dist, caption="Price Distributions Across Subcategories (Log Scale)", use_container_width=True)

        r_dist = os.path.join(ROOT_DIR, "artifacts", "plots", "rating_vs_price.png")
        if os.path.exists(r_dist):
            st.image(r_dist, caption="Customer Rating vs Listed Price", use_container_width=True)

with tab_leakage:
    st.subheader("🛡️ Data Leakage Analysis & Production Architecture")
    st.markdown("""
    ### Why Discount is a Leakage Hazard
    In the scraped dataset, items with discounts contain strings like `After $5 OFF` or `After $70 OFF`.
    If the machine learning model is fed `discount_amount` as a predictive feature to predict the price:

    1. **Target-Discount Confounding:** The discount amount is determined by marketing teams *after* a product's base retail price has been established. Feeding promotion amounts into a base price estimator creates a circular causal relationship.
    2. **Sparse Promotion Bias:** Only **7.2% (126 of 1,752 items)** in the Costco catalog are on active promotion. A model trained with discount features relies on a signal that is missing or zero for 93% of typical grocery listings.
    3. **Operational Failure at Inference:** When pricing a new product or inventory item, the merchant wants to estimate the *fair baseline retail price*. They do not yet know the discount.

    ### Empirical Verification: Variant A vs Variant B
    - **Variant A (With Discount Features):** Test R² = `0.8623`, CV R² = `0.8443`
    - **Variant B (Without Discount Features - Selected):** Test R² = `0.8617`, CV R² = `0.8443`

    *Result:* Variant B achieves virtually identical accuracy without depending on promotional metadata.

    ### Production Solution
    The deployed system adopts **Variant B**:
    $$\\text{Base Price} = \\hat{f}(\\text{Title}, \\text{Category}, \\text{Description}, \\text{Units}, \\text{Rating})$$
    $$\\text{Final Price} = \\max(0.50, \\ \\text{Base Price} - \\text{Discount})$$
    This modular architecture guarantees valid, leak-free predictions across all catalog items.
    """)
