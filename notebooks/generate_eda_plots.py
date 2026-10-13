import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("artifacts/plots", exist_ok=True)
os.makedirs("notebooks/eda_plots", exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

df = pd.read_csv("data/costco_grocery.csv")

def parse_price(val):
    if pd.isnull(val):
        return np.nan
    s = str(val).strip().replace(',', '')
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
    if len(nums) == 0:
        return np.nan
    elif len(nums) == 1:
        return nums[0]
    else:
        return sum(nums[:2]) / 2.0

def parse_discount(val):
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
    if pd.isnull(val):
        return np.nan, 0
    s = str(val).strip()
    match = re.search(r"Rated\s+([0-9.]+)\s+out of 5 stars based on\s+([0-9]+)\s+reviews", s)
    if match:
        return float(match.group(1)), int(match.group(2))
    return np.nan, 0

# Clean data
df['Price_clean'] = df['Price'].apply(parse_price)
# Drop rows without price or with scraper error 'Page Not Found!'
df_clean = df[df['Price_clean'].notnull() & (~df['Title'].str.contains("Page Not Found", case=False, na=False))].copy()

discount_tuples = df_clean['Discount'].apply(parse_discount)
df_clean['discount_amount'] = [t[0] for t in discount_tuples]
df_clean['discount_percent'] = [t[1] for t in discount_tuples]
df_clean['has_discount'] = [t[2] for t in discount_tuples]
mask_disc = (df_clean['has_discount'] == 1) & (df_clean['Price_clean'] > 0)
df_clean.loc[mask_disc, 'discount_percent'] = (df_clean.loc[mask_disc, 'discount_amount'] / df_clean.loc[mask_disc, 'Price_clean']) * 100.0

rating_tuples = df_clean['Rating'].apply(parse_rating)
df_clean['rating_clean'] = [t[0] for t in rating_tuples]
df_clean['review_count'] = [t[1] for t in rating_tuples]
df_clean['log_price'] = np.log1p(df_clean['Price_clean'])

print(f"Cleaned dataset rows: {len(df_clean)}")

# --- PLOT 1: Price Distribution (Raw & Log) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw Price
sns.histplot(df_clean['Price_clean'], bins=40, kde=True, ax=axes[0], color='#1f77b4')
axes[0].set_title('Raw Price Distribution ($)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Product Count')
axes[0].axvline(df_clean['Price_clean'].median(), color='red', linestyle='--', label=f'Median: ${df_clean["Price_clean"].median():.2f}')
axes[0].axvline(df_clean['Price_clean'].mean(), color='orange', linestyle=':', label=f'Mean: ${df_clean["Price_clean"].mean():.2f}')
axes[0].legend()

# Log Price
sns.histplot(df_clean['log_price'], bins=35, kde=True, ax=axes[1], color='#2ca02c')
axes[1].set_title('Log-Transformed Price log1p(Price)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log1p(Price)')
axes[1].set_ylabel('Product Count')
axes[1].axvline(df_clean['log_price'].median(), color='red', linestyle='--', label=f'Median: {df_clean["log_price"].median():.2f}')
axes[1].legend()

plt.tight_layout()
fig.savefig('artifacts/plots/price_distribution.png', dpi=300)
fig.savefig('notebooks/eda_plots/price_distribution.png', dpi=300)
plt.close(fig)
print("Saved price_distribution.png")

# --- PLOT 2: Price by Sub Category ---
plt.figure(figsize=(12, 8))
cat_order = df_clean.groupby('Sub Category')['Price_clean'].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x='Price_clean', y='Sub Category', order=cat_order, palette='viridis', showfliers=False)
plt.xscale('log')
plt.title('Costco Grocery Price Distribution by Sub Category (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Price ($) - Log Scale')
plt.ylabel('Sub Category')
plt.tight_layout()
plt.savefig('artifacts/plots/price_by_subcategory.png', dpi=300)
plt.savefig('notebooks/eda_plots/price_by_subcategory.png', dpi=300)
plt.close()
print("Saved price_by_subcategory.png")

# --- PLOT 3: Discount vs Price ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Discounted vs Non-Discounted Price boxplot
sns.boxplot(data=df_clean, x='has_discount', y='Price_clean', ax=axes[0], palette=['#4c72b0', '#55a868'], showfliers=False)
axes[0].set_xticklabels(['No Promo (0)', 'Has Promo (1)'])
axes[0].set_title('Price Distribution: Standard vs Promo Items', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Promotion Status')
axes[0].set_ylabel('Price ($)')

# Scatter Discount Amount vs Price for promo items
promo_df = df_clean[df_clean['has_discount'] == 1]
sns.scatterplot(data=promo_df, x='Price_clean', y='discount_amount', ax=axes[1], hue='Sub Category', s=80, alpha=0.8, legend=False)
axes[1].set_title('Discount Amount ($) vs Listed Price ($)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Price ($)')
axes[1].set_ylabel('Discount Amount ($)')

plt.tight_layout()
fig.savefig('artifacts/plots/discount_vs_price.png', dpi=300)
fig.savefig('notebooks/eda_plots/discount_vs_price.png', dpi=300)
plt.close(fig)
print("Saved discount_vs_price.png")

# --- PLOT 4: Rating vs Price ---
plt.figure(figsize=(10, 6))
rated_df = df_clean[df_clean['rating_clean'].notnull()].copy()
sns.regplot(data=rated_df, x='rating_clean', y='Price_clean', scatter_kws={'alpha': 0.4, 'color': '#1f77b4'}, line_kws={'color': 'crimson'})
plt.yscale('log')
plt.title('Customer Rating vs Price (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Rating (Stars out of 5)')
plt.ylabel('Price ($) - Log Scale')
corr = rated_df['rating_clean'].corr(rated_df['log_price'])
plt.annotate(f'Pearson r: {corr:.3f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
plt.tight_layout()
plt.savefig('artifacts/plots/rating_vs_price.png', dpi=300)
plt.savefig('notebooks/eda_plots/rating_vs_price.png', dpi=300)
plt.close()
print("Saved rating_vs_price.png")
