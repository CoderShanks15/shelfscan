"""
food_pipeline_final.py
======================
ShelfScan — Final Data Preparation Pipeline

Base : food_pipeline_v5_final.py  (structure + bug fixes)
Adds : safe features from v4 + v7  (no new bugs introduced)

What was added from v4:
  + traces_count
  + additives_animal_derived flag
  + sodium_salt_ratio
  + energy_density
  + high_fat_low_quality interaction
  + data_completeness filter
  + expanded HIGH_RISK / MEDIUM_RISK additive dicts
  + 8 more subcategories
  + 3 more label flags (is_no_salt, is_raw, is_low_sugar)

What was kept from v5 (unchanged):
  + Early sampling before all heavy processing
  + ingredients_text kept inside df (no alignment bugs)
  + Median-based energy unit detection
  + Salt mismatch filter
  + NOVA score + processing_score
  + All category + label one-hots
  + All interaction features
  + Improved TF-IDF token_pattern
  + fruits_veg_nuts.fillna(0) fix

What was NOT taken (intentionally skipped):
  - Fatty acid columns   (>95% null in OFF data)
  - glycemic_index       (>98% null, fillna(50) is fake data)
  - carbon_footprint     (<5% filled, near-useless feature)
  - pd.get_dummies       (produces inconsistent columns across runs)
  - v4 ingredients_raw   (alignment bug — v5 approach is correct)

Run:
  python data/food_pipeline_final.py

Outputs:
  data/cleaned_food_data_v4.csv
  data/tfidf_vectorizer.pkl
  data/tfidf_matrix.npz
  data/feature_cols.txt
"""

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
INPUT_PATH    = 'data/en.openfoodfacts.org.products.tsv'
OUTPUT_CSV    = 'data/cleaned_food_data_v4.csv'
TFIDF_PATH    = 'data/tfidf_vectorizer.pkl'
FEATURES_PATH = 'data/feature_cols.txt'
TFIDF_NPZ     = 'data/tfidf_matrix.npz'
SAMPLE_SIZE   = 150_000
RANDOM_STATE  = 42

os.makedirs('data', exist_ok=True)

# -----------------------------------------------------------------------
# STEP 1 — LOAD
# Only columns we actually use. Keeping memory low on the full 2GB TSV.
# 'traces' added vs v5 — needed for traces_count feature.
# -----------------------------------------------------------------------
COLUMNS = [
    'code', 'product_name', 'nutrition_grade_fr',
    'pnns_groups_1', 'pnns_groups_2', 'labels_tags',
    'energy_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g', 'sodium_100g',
    'trans-fat_100g', 'monounsaturated-fat_100g',
    'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'omega-6-fat_100g',
    'vitamin-a_100g', 'vitamin-c_100g', 'vitamin-d_100g',
    'vitamin-e_100g', 'vitamin-b1_100g', 'vitamin-b2_100g',
    'vitamin-b6_100g', 'vitamin-b9_100g', 'vitamin-b12_100g',
    'calcium_100g', 'iron_100g', 'magnesium_100g',
    'potassium_100g', 'zinc_100g',
    'fruits-vegetables-nuts_100g',
    'additives_n', 'ingredients_from_palm_oil_n',
    'ingredients_that_may_be_from_palm_oil_n',
    'allergens', 'traces',                          # traces added from v4
    'additives_tags', 'ingredients_text',
    'nutrition-score-fr_100g',
]

print("=" * 60)
print("SHELFSCAN — FOOD PIPELINE (FINAL)")
print("=" * 60)
print("\n[1/9] Loading dataset...")

df = pd.read_csv(
    INPUT_PATH, sep='\t', low_memory=False,
    usecols=COLUMNS, dtype={'code': 'str'}
)
print(f"  Raw rows: {len(df):,}")

# -----------------------------------------------------------------------
# STEP 2 — RENAME
# -----------------------------------------------------------------------
df = df.rename(columns={
    'nutrition_grade_fr':                       'nutriscore',
    'pnns_groups_1':                            'food_category',
    'pnns_groups_2':                            'food_subcategory',
    'labels_tags':                              'labels',
    'energy_100g':                              'energy_raw',
    'fat_100g':                                 'fat',
    'saturated-fat_100g':                       'saturated_fat',
    'trans-fat_100g':                           'trans_fat',
    'monounsaturated-fat_100g':                 'mono_fat',
    'polyunsaturated-fat_100g':                 'poly_fat',
    'omega-3-fat_100g':                         'omega3',
    'omega-6-fat_100g':                         'omega6',
    'carbohydrates_100g':                       'carbohydrates',
    'sugars_100g':                              'sugars',
    'fiber_100g':                               'fiber',
    'proteins_100g':                            'proteins',
    'salt_100g':                                'salt',
    'sodium_100g':                              'sodium',
    'vitamin-a_100g':                           'vitamin_a',
    'vitamin-c_100g':                           'vitamin_c',
    'vitamin-d_100g':                           'vitamin_d',
    'vitamin-e_100g':                           'vitamin_e',
    'vitamin-b1_100g':                          'vitamin_b1',
    'vitamin-b2_100g':                          'vitamin_b2',
    'vitamin-b6_100g':                          'vitamin_b6',
    'vitamin-b9_100g':                          'vitamin_b9',
    'vitamin-b12_100g':                         'vitamin_b12',
    'calcium_100g':                             'calcium',
    'iron_100g':                                'iron',
    'magnesium_100g':                           'magnesium',
    'potassium_100g':                           'potassium',
    'zinc_100g':                                'zinc',
    'fruits-vegetables-nuts_100g':              'fruits_veg_nuts',
    'additives_n':                              'additives_count',
    'ingredients_from_palm_oil_n':              'palm_oil_count',
    'ingredients_that_may_be_from_palm_oil_n':  'palm_oil_maybe_count',
    'nutrition-score-fr_100g':                  'nutrition_score',
})

# -----------------------------------------------------------------------
# STEP 3 — ENERGY UNIT FIX
# OFF stores energy in kJ. Median > 500 reliably identifies kJ columns.
# Row-by-row threshold (v4) was fragile — median detection is robust.
# -----------------------------------------------------------------------
median_energy = df['energy_raw'].median()
if median_energy > 500:
    df['energy_kcal'] = df['energy_raw'] / 4.184
    print(f"  Energy: median={median_energy:.0f} → detected kJ, converting")
else:
    df['energy_kcal'] = df['energy_raw']
    print(f"  Energy: median={median_energy:.0f} → already kcal")
df = df.drop(columns=['energy_raw'])

# -----------------------------------------------------------------------
# STEP 4 — CLEAN & FILTER
# -----------------------------------------------------------------------
print("\n[2/9] Cleaning...")

required_cols = [
    'energy_kcal', 'fat', 'saturated_fat',
    'carbohydrates', 'sugars', 'proteins', 'nutrition_score',
]
df = df.dropna(subset=required_cols)
print(f"  After dropping missing core nutrients : {len(df):,}")

df = df[
    (df['energy_kcal'].between(0, 900)) &
    (df['fat'].between(0, 100)) &
    (df['saturated_fat'].between(0, 100)) &
    (df['carbohydrates'].between(0, 100)) &
    (df['sugars'].between(0, 100)) &
    (df['proteins'].between(0, 100))
]
print(f"  After range filter                    : {len(df):,}")

# Salt mismatch filter — sodium * 2.5 should roughly equal salt.
# Removes rows where the two fields are wildly inconsistent (data corruption).
df['expected_salt'] = df['sodium'].fillna(0) * 2.5
df = df[(df['salt'].fillna(0) - df['expected_salt']).abs() < 5]
df = df.drop(columns=['expected_salt'])
print(f"  After salt mismatch filter            : {len(df):,}")

# Data completeness filter (from v4).
# Removes rows where fewer than 50% of core nutrients are nonzero.
# Catches products with mostly-empty nutrition panels.
core = ['fat', 'proteins', 'carbohydrates', 'fiber', 'sugars', 'salt']
df['data_completeness'] = df[core].gt(0).sum(axis=1) / len(core)
df = df[df['data_completeness'] >= 0.5]
df = df.drop(columns=['data_completeness'])   # drop — not a feature
print(f"  After completeness filter             : {len(df):,}")

df = df.reset_index(drop=True)

# -----------------------------------------------------------------------
# STEP 5 — EARLY SAMPLING
# Sample BEFORE all heavy processing — major speed gain over v4.
# ingredients_text stays inside df so alignment is guaranteed.
# -----------------------------------------------------------------------
print(f"\n[3/9] Sampling {SAMPLE_SIZE:,} rows before processing...")
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
df = df.reset_index(drop=True)
print(f"  Working set: {len(df):,} rows")

# -----------------------------------------------------------------------
# STEP 6 — CATEGORY & LABEL ONE-HOT ENCODING
# Explicit named maps (NOT pd.get_dummies) — always produces the same
# fixed column set regardless of what categories appear in the sample.
# Consistent schema is required for a fixed inference feature vector.
# -----------------------------------------------------------------------
print("\n[4/9] Encoding categories and labels...")

df['food_category'] = df['food_category'].fillna('unknown').str.strip()
cat_map = {
    'cat_beverages':      'Beverages',
    'cat_dairy':          'Milk and dairy products',
    'cat_meat_fish_eggs': 'Fish Meat Eggs',
    'cat_fruits_veg':     'Fruits and vegetables',
    'cat_sugary_snacks':  'Sugary snacks',
    'cat_salty_snacks':   'Salty snacks',
    'cat_cereals':        'Cereals and potatoes',
    'cat_fat_sauces':     'Fat and sauces',
    'cat_composite':      'Composite foods',
}
cat_cols = {
    col: (df['food_category'] == val).astype(int)
    for col, val in cat_map.items()
}

df['food_subcategory'] = df['food_subcategory'].fillna('unknown').str.strip()
# Expanded subcategory map — 26 entries (v4 level coverage)
subcat_map = {
    'subcat_biscuits_cakes':  'Biscuits and cakes',
    'subcat_chocolate':       'Chocolate products',
    'subcat_sweets':          'Sweets',
    'subcat_bread':           'Bread',
    'subcat_cereals':         'Cereals',
    'subcat_cheese':          'Cheese',
    'subcat_milk_yogurt':     'Milk and yogurt',
    'subcat_meat':            'Meat',
    'subcat_vegetables':      'Vegetables',
    'subcat_fruit_juices':    'Fruit juices',
    'subcat_sweetened_bev':   'Sweetened beverages',
    'subcat_non_sugared_bev': 'Non-sugared beverages',
    'subcat_alcoholic_bev':   'Alcoholic beverages',  # added from v4
    'subcat_legumes':         'Legumes',
    'subcat_pizza':           'Pizza pies and quiche', # added from v4
    'subcat_dressings':       'Dressings and sauces',  # added from v4
    'subcat_appetizers':      'Appetizers',            # added from v4
    'subcat_one_dish':        'One-dish meals',        # added from v4
    'subcat_sandwich':        'Sandwich',              # added from v4
    'subcat_fish':            'Fish and seafood',
    'subcat_eggs':            'Eggs',                  # added from v4
    'subcat_soups':           'Soups',
    'subcat_nuts':            'Nuts',
    'subcat_ice_cream':       'Ice cream',
    'subcat_pasta':           'Pasta',
    'subcat_rice':            'Rice and grains',       # added from v4
}
subcat_cols = {
    col: (df['food_subcategory'] == val).astype(int)
    for col, val in subcat_map.items()
}

# Expanded label map — 16 flags (v4 level coverage)
df['labels'] = df['labels'].fillna('')
label_map = {
    'is_organic':      'organic',
    'is_vegan':        'vegan',
    'is_vegetarian':   'vegetarian',
    'is_gluten_free':  'gluten-free|no-gluten',
    'is_fair_trade':   'fair-trade',
    'is_no_additives': 'no-additives',
    'is_low_sugar':    'low-sugar|no-sugar',           # added from v4
    'is_high_fiber':   'high-fiber|source-of-fiber',
    'is_no_salt':      'no-salt|low-salt',             # added from v4
    'is_wholegrain':   'whole-grain|wholegrain',
    'is_no_palm_oil':  'no-palm-oil',
    'is_halal':        'halal',
    'is_kosher':       'kosher',
    'is_raw':          'raw',                          # added from v4
    'is_fortified':    'fortified|enriched',
}
label_cols = {
    col: df['labels'].str.contains(pattern, case=False).astype(int)
    for col, pattern in label_map.items()
}

df = df.drop(columns=['food_category', 'food_subcategory', 'labels'])

# -----------------------------------------------------------------------
# STEP 7 — ADDITIVE & ALLERGEN SCORING
# Expanded HIGH_RISK and MEDIUM_RISK dicts from v4.
# All built as local Series before concat — fixes v4's KeyError bug
# where df['additive_risk_score'] was referenced before it existed.
# Animal-derived flag added from v4.
# -----------------------------------------------------------------------
print("\n[5/9] Scoring additives and allergens...")

# Expanded from v5 with additional E-numbers from v4
HIGH_RISK = {
    'en:e250': 5, 'en:e251': 5, 'en:e252': 5,
    'en:e102': 4, 'en:e110': 4, 'en:e124': 4, 'en:e129': 4,
    'en:e104': 3, 'en:e122': 3, 'en:e407': 3,  # added from v4
    'en:e621': 3, 'en:e951': 3, 'en:e954': 3,
    'en:e955': 3,                                # added from v4
    'en:e210': 3, 'en:e211': 3,
    'en:e150c': 2, 'en:e150d': 2,               # added from v4
}
MEDIUM_RISK = {
    'en:e320': 2, 'en:e321': 2,
    'en:e338': 2, 'en:e339': 2, 'en:e340': 1,  # added from v4
    'en:e450': 1, 'en:e451': 1, 'en:e452': 1,
    'en:e460': 1, 'en:e461': 1, 'en:e553b': 1, # added from v4
    'en:e171': 1, 'en:e172': 1,
    'en:e231': 2, 'en:e232': 2,
    'en:e310': 2, 'en:e311': 2, 'en:e312': 2,  # added from v4
    'en:e900': 1, 'en:e960': 1,                 # added from v4
}
# Animal-derived additives flag — added from v4
ANIMAL_DERIVED = [
    'en:e120', 'en:e441', 'en:e542',
    'en:e901', 'en:e904', 'en:e913', 'en:e966',
]


def score_additives(s):
    """Returns (high_risk_count, medium_risk_count, animal_count, risk_score)."""
    if not s or str(s) == 'nan':
        return 0, 0, 0, 0
    tags   = [t.strip().lower() for t in str(s).split(',')]
    high   = sum(1 for t in tags if t in HIGH_RISK)
    medium = sum(1 for t in tags if t in MEDIUM_RISK)
    animal = sum(1 for t in tags if t in ANIMAL_DERIVED)  # from v4
    risk   = (sum(HIGH_RISK.get(t, 0) for t in tags) +
              sum(MEDIUM_RISK.get(t, 0) for t in tags))
    return high, medium, animal, risk


add_results = df['additives_tags'].apply(score_additives)

# Local Series — NOT yet in df. No KeyError risk.
additive_high_risk    = add_results.apply(lambda x: x[0])
additive_medium_risk  = add_results.apply(lambda x: x[1])
additive_animal       = add_results.apply(lambda x: x[2])  # from v4
additive_risk_score   = add_results.apply(lambda x: x[3])

# Allergen + traces counts (traces_count added from v4)
allergens_count = df['allergens'].fillna('').apply(
    lambda x: len(str(x).split(',')) if x and str(x) != 'nan' else 0
)
traces_count = df['traces'].fillna('').apply(         # from v4
    lambda x: len(str(x).split(',')) if x and str(x) != 'nan' else 0
)

palm_oil_risk = (df['palm_oil_count'].fillna(0) +
                 df['palm_oil_maybe_count'].fillna(0) * 0.5)

df = df.drop(columns=['additives_tags', 'allergens', 'traces'])

# -----------------------------------------------------------------------
# STEP 8 — INGREDIENT PARSING (vectorized regex — no row loops)
# -----------------------------------------------------------------------
print("\n[6/9] Parsing ingredients (vectorized)...")

CONCERNING_RE = re.compile(
    r'hydrogenated|artificial\s+flavou?r|corn\s+syrup|'
    r'high\s+fructose|sodium\s+nitrite|sodium\s+nitrate|'
    r'aspartame|acesulfame|sucralose|carrageenan|'
    r'monosodium\s+glutamate|\bmsg\b|polysorbate',
    re.IGNORECASE
)
HEALTHY_RE = re.compile(
    r'whole[\-\s]?grain|whole[\-\s]?wheat|olive\s+oil|flaxseed|chia|'
    r'quinoa|spinach|kale|blueberr|almond|walnut|lentil|chickpea|'
    r'probiotic|lactobacillus|dark\s+chocolate|cacao|oats',
    re.IGNORECASE
)
SUGAR_TOP3_RE = re.compile(
    r'sugar|glucose|fructose|dextrose|corn\s+syrup|honey|syrup',
    re.IGNORECASE
)
FAT_TOP3_RE  = re.compile(r'\boil\b|fat|butter|cream|lard|margarine', re.IGNORECASE)
SALT_TOP3_RE = re.compile(r'\bsalt\b|sodium|\bmsg\b', re.IGNORECASE)

ingr = df['ingredients_text'].fillna('')
top3 = ingr.str.split(',').apply(
    lambda x: ' '.join(x[:3]) if isinstance(x, list) else ''
)

concerning_count = ingr.apply(lambda x: len(CONCERNING_RE.findall(x)))
healthy_count    = ingr.apply(lambda x: len(HEALTHY_RE.findall(x)))
has_concerning   = (concerning_count > 0).astype(int)
has_healthy      = (healthy_count > 0).astype(int)
sugar_in_top3    = top3.str.contains(SUGAR_TOP3_RE).astype(int)
fat_in_top3      = top3.str.contains(FAT_TOP3_RE).astype(int)
salt_in_top3     = top3.str.contains(SALT_TOP3_RE).astype(int)
ingredient_count = ingr.str.count(',') + 1
ultra_proc = (
    (ingredient_count > 10) |
    ingr.str.contains(r'hydrogenated|artificial', case=False, regex=True)
).astype(int)

# -----------------------------------------------------------------------
# STEP 9 — DERIVED FEATURES
# All computed from local variables — no df column references before concat.
# -----------------------------------------------------------------------
print("\n[7/9] Engineering derived features...")

additives_count_s = df['additives_count'].fillna(0)

# Processing score and NOVA — high-value signals, kept from v5
processing_score = (
    additives_count_s * 2 +
    ultra_proc * 3 +
    concerning_count * 2 +
    ingredient_count / 10
)
nova_score = pd.cut(
    processing_score,
    bins=[-0.1, 4, 8, 15, 9999],
    labels=[1, 2, 3, 4]
).astype('Int64').fillna(2).astype(int)

# Ratio features
fiber_sugar_ratio    = df['fiber'].fillna(0) / (df['sugars'].fillna(0) + 0.1)
protein_fat_ratio    = df['proteins'].fillna(0) / (df['fat'].fillna(0) + 0.1)
protein_energy_ratio = df['proteins'].fillna(0) / (df['energy_kcal'] + 1)
fat_quality_ratio    = (
    df['mono_fat'].fillna(0) + df['poly_fat'].fillna(0)
) / (df['saturated_fat'].fillna(0) + 0.1)
omega_balance      = df['omega3'].fillna(0) / (df['omega6'].fillna(0) + 0.001)
carb_fiber_balance = df['carbohydrates'] / (df['fiber'].fillna(0) + 0.1)
is_high_energy     = (df['energy_kcal'] > 400).astype(int)

# Added from v4 — safe standalone derived features
energy_density     = df['energy_kcal'] / 100
sodium_salt_ratio  = df['sodium'].fillna(0) / (df['salt'].fillna(0) + 0.001)
high_fat_low_quality = (                          # from v4
    df['fat'].fillna(0) * (1 / (fat_quality_ratio + 0.1))
)

# Interaction features
sugar_AND_no_fiber        = df['sugars'].fillna(0) * (1 / (df['fiber'].fillna(0) + 0.1))
additive_AND_processed    = additive_risk_score * ultra_proc
salt_AND_fat_bomb         = (
    (df['salt'].fillna(0) > 2) & (df['fat'].fillna(0) > 20)
).astype(int)
protein_fiber_bonus       = df['proteins'].fillna(0) * df['fiber'].fillna(0)
sugar_AND_ultra_processed = df['sugars'].fillna(0) * ultra_proc
omega3_sat_ratio          = df['omega3'].fillna(0) / (df['saturated_fat'].fillna(0) + 0.1)

# whole_food_score — uses is_wholegrain from label_cols (already computed)
whole_food_score = (
    df['fiber'].fillna(0) * label_cols['is_wholegrain'] /
    (additives_count_s + ultra_proc + 1)
)

# [FIX] fruits_veg_nuts.fillna(0) — column is ~80% null in OFF data
macro_balance_score = (
    (df['proteins'].fillna(0) * 4 +
     df['fiber'].fillna(0) * 2 +
     df['fruits_veg_nuts'].fillna(0) * 1.5) /
    (df['sugars'].fillna(0) * 3 +
     df['saturated_fat'].fillna(0) * 2 +
     df['salt'].fillna(0) * 5 + 1)
)

# Vitamin density — sum of % daily reference values per 100g
VITAMIN_REF = {
    'vitamin_a': 900, 'vitamin_c': 90,  'vitamin_d': 20,
    'vitamin_e': 15,  'vitamin_b1': 1.2, 'vitamin_b2': 1.3,
    'vitamin_b6': 1.7, 'vitamin_b9': 400, 'vitamin_b12': 2.4,
}
vit_score = sum(
    df[v].fillna(0) / ref
    for v, ref in VITAMIN_REF.items() if v in df.columns
)
mineral_density = (
    df[['calcium', 'iron', 'magnesium', 'potassium', 'zinc']]
    .fillna(0).sum(axis=1)
)

# -----------------------------------------------------------------------
# STEP 10 — CONCAT ALL NEW COLUMNS AT ONCE
# Single pd.concat = zero DataFrame fragmentation.
# -----------------------------------------------------------------------
print("\n[8/9] Assembling final dataframe...")

df = df.drop(columns=['palm_oil_count', 'palm_oil_maybe_count', 'additives_count'])

derived_df = pd.DataFrame({
    # Processing signals
    'processing_score':             processing_score,
    'nova_score':                   nova_score,

    # Ratio features
    'fiber_sugar_ratio':            fiber_sugar_ratio,
    'protein_fat_ratio':            protein_fat_ratio,
    'protein_energy_ratio':         protein_energy_ratio,
    'fat_quality_ratio':            fat_quality_ratio,
    'omega_balance':                omega_balance,
    'carb_fiber_balance':           carb_fiber_balance,

    # Density features
    'energy_density':               energy_density,        # from v4
    'sodium_salt_ratio':            sodium_salt_ratio,     # from v4
    'is_high_energy_density':       is_high_energy,

    # Interaction features
    'sugar_AND_no_fiber':           sugar_AND_no_fiber,
    'high_fat_low_quality':         high_fat_low_quality,  # from v4
    'additive_AND_processed':       additive_AND_processed,
    'salt_AND_fat_bomb':            salt_AND_fat_bomb,
    'protein_fiber_bonus':          protein_fiber_bonus,
    'sugar_AND_ultra_processed':    sugar_AND_ultra_processed,
    'omega3_sat_ratio':             omega3_sat_ratio,
    'whole_food_score':             whole_food_score,
    'macro_balance_score':          macro_balance_score,

    # Micronutrient density
    'vitamin_density_score':        vit_score,
    'mineral_density_score':        mineral_density,

    # Palm oil and additives
    'palm_oil_risk':                palm_oil_risk,
    'additives_high_risk':          additive_high_risk,
    'additives_medium_risk':        additive_medium_risk,
    'additives_animal_derived':     additive_animal,       # from v4
    'additive_risk_score':          additive_risk_score,

    # Allergens
    'allergens_count':              allergens_count,
    'traces_count':                 traces_count,          # from v4

    # Ingredient signals
    'concerning_ingredient_count':  concerning_count,
    'healthy_ingredient_count':     healthy_count,
    'has_concerning_ingredient':    has_concerning,
    'has_healthy_ingredient':       has_healthy,
    'sugar_in_top3':                sugar_in_top3,
    'fat_in_top3':                  fat_in_top3,
    'salt_in_top3':                 salt_in_top3,
    'ingredient_count':             ingredient_count,
    'ultra_processed_indicator':    ultra_proc,
}, index=df.index)

df = pd.concat([
    df,
    pd.DataFrame(cat_cols,    index=df.index),
    pd.DataFrame(subcat_cols, index=df.index),
    pd.DataFrame(label_cols,  index=df.index),
    derived_df,
], axis=1)

# Remove any accidental duplicate columns (explicit, not silent)
df = df.loc[:, ~df.columns.duplicated(keep='first')]
print(f"  Assembled: {df.shape}")

# -----------------------------------------------------------------------
# BUILD TARGETS
# -----------------------------------------------------------------------
EXCLUDE_FILL = [
    'code', 'product_name', 'nutriscore',
    'nutrition_score', 'ingredients_text',
]
fill_cols = [c for c in df.columns if c not in EXCLUDE_FILL]
df[fill_cols] = df[fill_cols].fillna(0)
df['nutriscore'] = df['nutriscore'].fillna('unknown')

# Continuous target: 0 = worst, 100 = best
df['health_score'] = 100 - ((df['nutrition_score'] + 15) / 55 * 100)
df['health_score'] = df['health_score'].clip(0, 100)

# 3-class tier for classification tasks
df['health_tier'] = pd.cut(
    df['health_score'],
    bins=[0, 40, 70, 100],
    labels=[0, 1, 2],
    include_lowest=True,
).astype(int)

# -----------------------------------------------------------------------
# STEP 11 — TF-IDF
# token_pattern=r"[a-zA-Z\-]{2,}" captures hyphenated tokens correctly:
# soy-lecithin, palm-oil, sodium-benzoate — not split at the hyphen.
# ingredients_text still inside df — perfectly aligned, no tracking needed.
# -----------------------------------------------------------------------
print("\n[9/9] Fitting TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=250,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    token_pattern=r"[a-zA-Z\-]{2,}",
)
tfidf_matrix = tfidf.fit_transform(df['ingredients_text'].fillna(''))

assert tfidf_matrix.shape[0] == len(df), (
    f"SHAPE MISMATCH: tfidf={tfidf_matrix.shape[0]} csv={len(df)}"
)
print(f"  TF-IDF : {tfidf_matrix.shape} ✅   CSV rows: {len(df):,} ✅")

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
save_npz(TFIDF_NPZ, tfidf_matrix)
with open(TFIDF_PATH, 'wb') as f:
    pickle.dump(tfidf, f)

EXCLUDE_FEATURES = [
    'code', 'product_name', 'nutriscore', 'nutrition_score',
    'ingredients_text', 'health_score', 'health_tier',
]
feature_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES]
with open(FEATURES_PATH, 'w') as f:
    for i, col in enumerate(feature_cols):
        f.write(f"{i},{col}\n")

# -----------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("DATA PREPARATION COMPLETE")
print(f"{'=' * 60}")
print(f"  Rows                : {len(df):,}")
print(f"  Tabular features    : {len(feature_cols)}")
print(f"  TF-IDF features     : {tfidf_matrix.shape[1]}")
print(f"  Total features      : {len(feature_cols) + tfidf_matrix.shape[1]}")

print(f"\n  Health score distribution:")
print(df['health_score'].describe().round(2).to_string())

print(f"\n  Health tier distribution:")
for cls, name in [(0, 'Unhealthy'), (1, 'Moderate'), (2, 'Healthy')]:
    n = (df['health_tier'] == cls).sum()
    print(f"    {cls} ({name}): {n:,}  ({n / len(df) * 100:.1f}%)")

print(f"\n  Nutriscore breakdown:")
print(df['nutriscore'].value_counts().to_string())

print(f"\n  NOVA score distribution:")
print(df['nova_score'].value_counts().sort_index().to_string())

print(f"\n  Saved:")
print(f"    {OUTPUT_CSV}")
print(f"    {TFIDF_PATH}")
print(f"    {TFIDF_NPZ}")
print(f"    {FEATURES_PATH}")
print(f"{'=' * 60}")
print("\nNext: python ml/health_model.py")