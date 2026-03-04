"""
food_pipeline_v4.py
===================
Data preparation only.
Cleans raw OpenFoodFacts TSV -> saves cleaned_food_data_v4.csv

Run this once before training:
  python data/food_pipeline_v4.py

Output:
  data/cleaned_food_data_v4.csv   ← used by ml/health_model.py
  data/tfidf_vectorizer.pkl       ← saved for inference in app.py
  data/tfidf_matrix.npz           ← aligned with CSV rows
  data/feature_cols.txt           ← feature column names with index
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, csr_matrix

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
# -----------------------------------------------------------------------
COLUMNS = [
    'code', 'product_name', 'nutrition_grade_fr',
    'pnns_groups_1', 'pnns_groups_2',
    'labels_tags',
    'energy_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
    'proteins_100g', 'salt_100g', 'sodium_100g',
    'trans-fat_100g', 'monounsaturated-fat_100g',
    'polyunsaturated-fat_100g', 'omega-3-fat_100g',
    'omega-6-fat_100g', 'cholesterol_100g',
    '-butyric-acid_100g', '-caproic-acid_100g',
    '-caprylic-acid_100g', '-capric-acid_100g',
    '-lauric-acid_100g', '-myristic-acid_100g',
    '-palmitic-acid_100g', '-stearic-acid_100g',
    '-arachidic-acid_100g', '-behenic-acid_100g',
    '-lignoceric-acid_100g', '-cerotic-acid_100g',
    '-montanic-acid_100g', '-melissic-acid_100g',
    '-alpha-linolenic-acid_100g', '-eicosapentaenoic-acid_100g',
    '-docosahexaenoic-acid_100g', '-linoleic-acid_100g',
    '-arachidonic-acid_100g', '-gamma-linolenic-acid_100g',
    '-dihomo-gamma-linolenic-acid_100g', '-oleic-acid_100g',
    '-elaidic-acid_100g', '-gondoic-acid_100g',
    '-mead-acid_100g', '-erucic-acid_100g',
    '-nervonic-acid_100g',
    'starch_100g', 'polyols_100g',
    'vitamin-a_100g', 'beta-carotene_100g',
    'vitamin-c_100g', 'vitamin-d_100g',
    'vitamin-e_100g', 'vitamin-k_100g',
    'vitamin-b1_100g', 'vitamin-b2_100g',
    'vitamin-b6_100g', 'vitamin-b9_100g',
    'folates_100g', 'vitamin-b12_100g',
    'vitamin-pp_100g', 'biotin_100g',
    'pantothenic-acid_100g',
    'calcium_100g', 'iron_100g', 'magnesium_100g',
    'potassium_100g', 'zinc_100g', 'phosphorus_100g',
    'copper_100g', 'manganese_100g', 'fluoride_100g',
    'selenium_100g', 'chromium_100g', 'molybdenum_100g',
    'iodine_100g', 'silica_100g', 'bicarbonate_100g',
    'chloride_100g',
    'alcohol_100g', 'caffeine_100g', 'taurine_100g',
    'fruits-vegetables-nuts_100g',
    'fruits-vegetables-nuts-estimate_100g',
    'cocoa_100g', 'collagen-meat-protein-ratio_100g',
    'chlorophyl_100g', 'ph_100g',
    'glycemic-index_100g',
    'additives_n', 'ingredients_from_palm_oil_n',
    'ingredients_that_may_be_from_palm_oil_n',
    'allergens', 'traces',
    'ingredients_text',
    'additives_tags',
    'carbon-footprint_100g', 'water-hardness_100g',
    'nutrition-score-fr_100g',
]

print("=" * 60)
print("FOOD PIPELINE v4 — DATA PREPARATION")
print("=" * 60)
print("\n[1/10] Loading dataset...")

df = pd.read_csv(
    INPUT_PATH,
    sep='\t',
    low_memory=False,
    usecols=COLUMNS,
    dtype={'code': 'str'}
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
    'energy_100g':                              'energy_kcal',
    'fat_100g':                                 'fat',
    'saturated-fat_100g':                       'saturated_fat',
    'trans-fat_100g':                           'trans_fat',
    'monounsaturated-fat_100g':                 'mono_fat',
    'polyunsaturated-fat_100g':                 'poly_fat',
    'omega-3-fat_100g':                         'omega3',
    'omega-6-fat_100g':                         'omega6',
    'cholesterol_100g':                         'cholesterol',
    '-butyric-acid_100g':                       'butyric_acid',
    '-caproic-acid_100g':                       'caproic_acid',
    '-caprylic-acid_100g':                      'caprylic_acid',
    '-capric-acid_100g':                        'capric_acid',
    '-lauric-acid_100g':                        'lauric_acid',
    '-myristic-acid_100g':                      'myristic_acid',
    '-palmitic-acid_100g':                      'palmitic_acid',
    '-stearic-acid_100g':                       'stearic_acid',
    '-arachidic-acid_100g':                     'arachidic_acid',
    '-behenic-acid_100g':                       'behenic_acid',
    '-lignoceric-acid_100g':                    'lignoceric_acid',
    '-cerotic-acid_100g':                       'cerotic_acid',
    '-montanic-acid_100g':                      'montanic_acid',
    '-melissic-acid_100g':                      'melissic_acid',
    '-alpha-linolenic-acid_100g':               'alpha_linolenic_acid',
    '-eicosapentaenoic-acid_100g':              'epa',
    '-docosahexaenoic-acid_100g':               'dha',
    '-linoleic-acid_100g':                      'linoleic_acid',
    '-arachidonic-acid_100g':                   'arachidonic_acid',
    '-gamma-linolenic-acid_100g':               'gamma_linolenic_acid',
    '-dihomo-gamma-linolenic-acid_100g':        'dihomo_gamma_linolenic_acid',
    '-oleic-acid_100g':                         'oleic_acid',
    '-elaidic-acid_100g':                       'elaidic_acid',
    '-gondoic-acid_100g':                       'gondoic_acid',
    '-mead-acid_100g':                          'mead_acid',
    '-erucic-acid_100g':                        'erucic_acid',
    '-nervonic-acid_100g':                      'nervonic_acid',
    'carbohydrates_100g':                       'carbohydrates',
    'sugars_100g':                              'sugars',
    'starch_100g':                              'starch',
    'polyols_100g':                             'polyols',
    'fiber_100g':                               'fiber',
    'proteins_100g':                            'proteins',
    'salt_100g':                                'salt',
    'sodium_100g':                              'sodium',
    'alcohol_100g':                             'alcohol',
    'caffeine_100g':                            'caffeine',
    'taurine_100g':                             'taurine',
    'vitamin-a_100g':                           'vitamin_a',
    'beta-carotene_100g':                       'beta_carotene',
    'vitamin-c_100g':                           'vitamin_c',
    'vitamin-d_100g':                           'vitamin_d',
    'vitamin-e_100g':                           'vitamin_e',
    'vitamin-k_100g':                           'vitamin_k',
    'vitamin-b1_100g':                          'vitamin_b1',
    'vitamin-b2_100g':                          'vitamin_b2',
    'vitamin-b6_100g':                          'vitamin_b6',
    'vitamin-b9_100g':                          'vitamin_b9',
    'folates_100g':                             'folates',
    'vitamin-b12_100g':                         'vitamin_b12',
    'vitamin-pp_100g':                          'vitamin_pp',
    'biotin_100g':                              'biotin',
    'pantothenic-acid_100g':                    'pantothenic_acid',
    'calcium_100g':                             'calcium',
    'iron_100g':                                'iron',
    'magnesium_100g':                           'magnesium',
    'potassium_100g':                           'potassium',
    'zinc_100g':                                'zinc',
    'phosphorus_100g':                          'phosphorus',
    'copper_100g':                              'copper',
    'manganese_100g':                           'manganese',
    'fluoride_100g':                            'fluoride',
    'selenium_100g':                            'selenium',
    'chromium_100g':                            'chromium',
    'molybdenum_100g':                          'molybdenum',
    'iodine_100g':                              'iodine',
    'silica_100g':                              'silica',
    'bicarbonate_100g':                         'bicarbonate',
    'chloride_100g':                            'chloride',
    'fruits-vegetables-nuts_100g':              'fruits_veg_nuts',
    'fruits-vegetables-nuts-estimate_100g':     'fruits_veg_nuts_estimate',
    'cocoa_100g':                               'cocoa',
    'collagen-meat-protein-ratio_100g':         'collagen_ratio',
    'chlorophyl_100g':                          'chlorophyl',
    'ph_100g':                                  'ph',
    'glycemic-index_100g':                      'glycemic_index',
    'additives_n':                              'additives_count',
    'ingredients_from_palm_oil_n':              'palm_oil_count',
    'ingredients_that_may_be_from_palm_oil_n':  'palm_oil_maybe_count',
    'carbon-footprint_100g':                    'carbon_footprint',
    'water-hardness_100g':                      'water_hardness',
    'nutrition-score-fr_100g':                  'nutrition_score',
})

# -----------------------------------------------------------------------
# STEP 3 — ENERGY UNIT FIX
# OFF stores energy in kJ. Some rows may already be kcal.
# Only divide rows where value > 900 (impossible as kcal/100g)
# -----------------------------------------------------------------------
mask_kj = df['energy_kcal'] > 900
df.loc[mask_kj, 'energy_kcal'] = df.loc[mask_kj, 'energy_kcal'] / 4.184
print(f"\n  Energy unit fix: {mask_kj.sum():,} rows converted kJ → kcal")

# -----------------------------------------------------------------------
# STEP 4 — SAVE INGREDIENTS BEFORE ANY ROW DROPS
# -----------------------------------------------------------------------
ingredients_raw = df['ingredients_text'].copy().reset_index(drop=True)

# -----------------------------------------------------------------------
# STEP 5 — CLEAN
# -----------------------------------------------------------------------
print("\n[2/10] Cleaning data...")

required_cols = [
    'energy_kcal', 'fat', 'saturated_fat',
    'carbohydrates', 'sugars', 'fiber',
    'proteins', 'salt', 'nutrition_score'
]
df = df.dropna(subset=required_cols)
print(f"  After dropping missing: {len(df):,}")

df = df[
    (df['energy_kcal'].between(0, 900)) &
    (df['fat'].between(0, 100)) &
    (df['saturated_fat'].between(0, 100)) &
    (df['carbohydrates'].between(0, 100)) &
    (df['sugars'].between(0, 100)) &
    (df['proteins'].between(0, 100)) &
    (df['salt'].between(0, 100)) &
    (df['fiber'].between(0, 100))
]
print(f"  After impossible values: {len(df):,}")

# FIX: fillna(0) before salt mismatch filter
# prevents dropping rows where sodium OR salt is NaN
df['expected_salt'] = df['sodium'].fillna(0) * 2.5
df['salt_mismatch'] = (
    df['salt'].fillna(0) - df['expected_salt']).abs()
before = len(df)
df = df[df['salt_mismatch'].fillna(0) < 5]
print(f"  After salt filter: {len(df):,} (removed {before-len(df):,})")

core = ['fat', 'proteins', 'carbohydrates', 'fiber', 'sugars', 'salt']
df['data_completeness'] = df[core].gt(0).sum(axis=1) / len(core)
before = len(df)
df = df[df['data_completeness'] >= 0.5]
print(f"  After completeness filter: {len(df):,} (removed {before-len(df):,})")

# FIX: capture integer positions before reset for ingredients alignment
clean_positions = df.index.tolist()
df = df.reset_index(drop=True)
ingredients_clean = ingredients_raw.iloc[
    clean_positions].reset_index(drop=True)
print(f"  Index reset. Final rows: {len(df):,}")
assert len(ingredients_clean) == len(df), "Ingredients alignment failed"

# -----------------------------------------------------------------------
# STEP 6 — FOOD CATEGORY ONE-HOT
# FIX: collect all new columns as list then concat once (no fragmentation)
# -----------------------------------------------------------------------
print("\n[3/10] Encoding categories...")

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
df = df.drop(columns=['food_category'])

df['food_subcategory'] = df['food_subcategory'].fillna('unknown').str.strip()
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
    'subcat_alcoholic_bev':   'Alcoholic beverages',
    'subcat_legumes':         'Legumes',
    'subcat_pizza':           'Pizza pies and quiche',
    'subcat_dressings':       'Dressings and sauces',
    'subcat_appetizers':      'Appetizers',
    'subcat_one_dish':        'One-dish meals',
    'subcat_sandwich':        'Sandwich',
    'subcat_fish':            'Fish and seafood',
    'subcat_eggs':            'Eggs',
    'subcat_soups':           'Soups',
    'subcat_nuts':            'Nuts',
    'subcat_ice_cream':       'Ice cream',
    'subcat_pasta':           'Pasta',
    'subcat_rice':            'Rice and grains',
}
subcat_cols = {
    col: (df['food_subcategory'] == val).astype(int)
    for col, val in subcat_map.items()
}
df = df.drop(columns=['food_subcategory'])

# -----------------------------------------------------------------------
# STEP 7 — LABEL FLAGS
# -----------------------------------------------------------------------
df['labels'] = df['labels'].fillna('')
label_map = {
    'is_organic':       'organic',
    'is_vegan':         'vegan',
    'is_vegetarian':    'vegetarian',
    'is_gluten_free':   'gluten-free|no-gluten',
    'is_fair_trade':    'fair-trade',
    'is_no_additives':  'no-additives',
    'is_low_sugar':     'low-sugar|no-sugar',
    'is_high_fiber':    'high-fiber|source-of-fiber',
    'is_no_salt':       'no-salt|low-salt',
    'is_wholegrain':    'whole-grain|wholegrain',
    'is_no_palm_oil':   'no-palm-oil',
    'is_halal':         'halal',
    'is_kosher':        'kosher',
    'is_raw':           'raw',
    'is_fortified':     'fortified|enriched',
}
label_cols = {
    col: df['labels'].str.contains(pattern, case=False).astype(int)
    for col, pattern in label_map.items()
}
df = df.drop(columns=['labels'])

# -----------------------------------------------------------------------
# STEP 8 — ALLERGENS + ADDITIVES
# -----------------------------------------------------------------------
print("\n[4/10] Scoring allergens and additives...")

allergen_cols = {
    'allergens_count': df['allergens'].fillna('').apply(
        lambda x: len(str(x).split(',')) if x and str(x) != 'nan' else 0
    ),
    'traces_count': df['traces'].fillna('').apply(
        lambda x: len(str(x).split(',')) if x and str(x) != 'nan' else 0
    ),
}
df = df.drop(columns=['allergens', 'traces'])

HIGH_RISK_ADDITIVES = {
    'en:e250': 5, 'en:e251': 5, 'en:e252': 5,
    'en:e102': 4, 'en:e110': 4, 'en:e124': 4, 'en:e129': 4,
    'en:e104': 3, 'en:e122': 3, 'en:e407': 3,
    'en:e621': 3, 'en:e951': 3, 'en:e954': 3, 'en:e955': 3,
    'en:e210': 3, 'en:e211': 3,
    'en:e150c': 2, 'en:e150d': 2,
}
MEDIUM_RISK_ADDITIVES = {
    'en:e320': 2, 'en:e321': 2,
    'en:e338': 2, 'en:e339': 2, 'en:e340': 1,
    'en:e450': 1, 'en:e451': 1, 'en:e452': 1,
    'en:e460': 1, 'en:e461': 1, 'en:e553b': 1,
    'en:e120': 2, 'en:e171': 1, 'en:e172': 1,
    'en:e231': 2, 'en:e232': 2,
    'en:e310': 2, 'en:e311': 2, 'en:e312': 2,
    'en:e900': 1, 'en:e960': 1,
}
ANIMAL_DERIVED_ADDITIVES = [
    'en:e120', 'en:e441', 'en:e542',
    'en:e901', 'en:e904', 'en:e913', 'en:e966',
]


def score_additives(additives_str):
    if not additives_str or str(additives_str) == 'nan':
        return 0, 0, 0, 0
    tags = [t.strip().lower() for t in str(additives_str).split(',')]
    high   = sum(1 for t in tags if t in HIGH_RISK_ADDITIVES)
    medium = sum(1 for t in tags if t in MEDIUM_RISK_ADDITIVES)
    animal = sum(1 for t in tags if t in ANIMAL_DERIVED_ADDITIVES)
    risk   = (
        sum(HIGH_RISK_ADDITIVES.get(t, 0) for t in tags) +
        sum(MEDIUM_RISK_ADDITIVES.get(t, 0) for t in tags)
    )
    return high, medium, animal, risk


additive_results = df['additives_tags'].apply(score_additives)
additive_cols = {
    'additives_high_risk':      additive_results.apply(lambda x: x[0]),
    'additives_medium_risk':    additive_results.apply(lambda x: x[1]),
    'additives_animal_derived': additive_results.apply(lambda x: x[2]),
    'additive_risk_score':      additive_results.apply(lambda x: x[3]),
}
df = df.drop(columns=['additives_tags'])

# -----------------------------------------------------------------------
# STEP 9 — INGREDIENT PARSING
# -----------------------------------------------------------------------
print("\n[5/10] Parsing ingredients...")

ANIMAL_FATS = [
    'lard', 'tallow', 'suet', 'ghee', 'dripping', 'butter fat',
    'beef fat', 'pork fat', 'chicken fat', 'duck fat', 'fish oil',
    'anchovy', 'animal fat', 'schmaltz', 'bacon fat',
]
ANIMAL_DERIVATIVES = [
    'gelatin', 'gelatine', 'rennet', 'casein', 'caseinate',
    'whey', 'lactose', 'carmine', 'cochineal', 'isinglass',
    'shellac', 'albumin', 'collagen', 'bone', 'lard',
    'tallow', 'suet', 'lanolin', 'beeswax',
]
CONCERNING_INGREDIENTS = [
    'high fructose corn syrup', 'hfcs', 'corn syrup',
    'hydrogenated', 'partially hydrogenated', 'interesterified',
    'modified starch', 'artificial flavour', 'artificial flavor',
    'artificial colour', 'artificial color',
    'monosodium glutamate', 'msg', 'sodium nitrite', 'sodium nitrate',
    'potassium bromate', 'bromated flour', 'propyl gallate', 'bha', 'bht',
    'carrageenan', 'polysorbate', 'sodium benzoate', 'potassium benzoate',
    'acesulfame', 'aspartame', 'saccharin', 'sucralose',
    'neotame', 'advantame',
]
HEALTHY_INGREDIENTS = [
    'whole grain', 'wholegrain', 'whole wheat', 'oats', 'quinoa',
    'flaxseed', 'chia', 'olive oil', 'avocado oil',
    'turmeric', 'ginger', 'garlic', 'spinach', 'kale', 'broccoli',
    'blueberr', 'strawberr', 'raspberry',
    'almond', 'walnut', 'cashew', 'pistachio',
    'lentil', 'chickpea', 'black bean',
    'probiotic', 'prebiotic', 'lactobacillus',
    'apple cider vinegar', 'green tea', 'dark chocolate', 'cacao',
]
SUGAR_WORDS = [
    'sugar', 'glucose', 'fructose', 'dextrose', 'sucrose', 'maltose',
    'corn syrup', 'hfcs', 'honey', 'invert sugar', 'molasses',
    'agave', 'cane juice', 'caramel', 'syrup',
]
FAT_WORDS = [
    'oil', 'fat', 'butter', 'cream', 'lard', 'tallow', 'ghee',
    'shortening', 'margarine', 'palm oil', 'coconut oil',
]
SALT_WORDS = ['salt', 'sodium', 'brine', 'msg']


def parse_ingredients(text):
    empty = {
        'has_animal_fat': 0, 'has_animal_derivative': 0,
        'has_concerning_ingredient': 0, 'concerning_ingredient_count': 0,
        'has_healthy_ingredient': 0, 'healthy_ingredient_count': 0,
        'sugar_in_top3': 0, 'fat_in_top3': 0, 'salt_in_top3': 0,
        'ingredient_count': 0, 'ultra_processed_indicator': 0,
    }
    if not text or str(text) == 'nan':
        return empty

    text_lower = str(text).lower()
    ingredients = [i.strip() for i in text_lower.split(',')]
    top3 = ' '.join(ingredients[:3])
    ultra = int(
        len(ingredients) > 10 or
        'hydrogenated' in text_lower or
        'artificial' in text_lower or
        'flavour' in text_lower or
        'flavor' in text_lower
    )
    return {
        'has_animal_fat':
            int(any(af in text_lower for af in ANIMAL_FATS)),
        'has_animal_derivative':
            int(any(ad in text_lower for ad in ANIMAL_DERIVATIVES)),
        'has_concerning_ingredient':
            int(any(ci in text_lower for ci in CONCERNING_INGREDIENTS)),
        'concerning_ingredient_count':
            sum(1 for ci in CONCERNING_INGREDIENTS if ci in text_lower),
        'has_healthy_ingredient':
            int(any(hi in text_lower for hi in HEALTHY_INGREDIENTS)),
        'healthy_ingredient_count':
            sum(1 for hi in HEALTHY_INGREDIENTS if hi in text_lower),
        'sugar_in_top3':
            int(any(w in top3 for w in SUGAR_WORDS)),
        'fat_in_top3':
            int(any(w in top3 for w in FAT_WORDS)),
        'salt_in_top3':
            int(any(w in top3 for w in SALT_WORDS)),
        'ingredient_count':
            len(ingredients),
        'ultra_processed_indicator':
            ultra,
    }


ingredient_results = df['ingredients_text'].apply(parse_ingredients)
ingredient_cols = pd.DataFrame(ingredient_results.tolist())
df = df.drop(columns=['ingredients_text'])

# -----------------------------------------------------------------------
# STEP 10 — ALL DERIVED FEATURES (collected, concat once)
# FIX: no individual column assignment in loops = no fragmentation
# -----------------------------------------------------------------------
print("\n[6/10] Engineering features...")

additive_count   = df['additives_count']
ultra_proc       = ingredient_cols['ultra_processed_indicator']
concerning_count = ingredient_cols['concerning_ingredient_count']
ingr_count       = ingredient_cols['ingredient_count']
is_wholegrain    = label_cols['is_wholegrain']

processing_score = (
    additive_count * 2 +
    ultra_proc * 3 +
    concerning_count * 2 +
    ingr_count / 10
)
nova_score = pd.cut(
    processing_score,
    bins=[-0.1, 4, 8, 15, 9999],
    labels=[1, 2, 3, 4]
).astype(int)

glycemic_index  = df['glycemic_index'].fillna(50)
glycemic_load   = glycemic_index * df['carbohydrates'] / 100

fiber_sugar_ratio    = df['fiber'] / (df['sugars'] + 0.1)
protein_fat_ratio    = df['proteins'] / (df['fat'] + 0.1)
protein_energy_ratio = df['proteins'] / (df['energy_kcal'] + 1)
fat_quality_ratio    = (
    df['mono_fat'] + df['poly_fat']) / (df['saturated_fat'] + 0.1)
omega_balance        = df['omega3'] / (df['omega6'] + 0.001)
carb_fiber_balance   = df['carbohydrates'] / (df['fiber'] + 0.1)
energy_density       = df['energy_kcal'] / 100
sodium_salt_ratio    = df['sodium'] / (df['salt'] + 0.001)
is_high_energy       = (df['energy_kcal'] > 400).astype(int)

sugar_AND_no_fiber        = df['sugars'] * (1 / (df['fiber'] + 0.1))
high_fat_low_quality      = df['fat'] * (1 / (fat_quality_ratio + 0.1))
additive_AND_processed    = df['additive_risk_score'].fillna(0) * ultra_proc
salt_AND_fat_bomb         = (
    (df['salt'] > 2).astype(int) * (df['fat'] > 20).astype(int))
protein_fiber_bonus       = df['proteins'] * df['fiber']
sugar_AND_ultra_processed = df['sugars'] * ultra_proc
omega3_sat_ratio          = df['omega3'] / (df['saturated_fat'] + 0.1)
whole_food_score          = (
    df['fiber'] * is_wholegrain /
    (additive_count + ultra_proc + 1)
)

macro_balance_score = (
    (df['proteins'] * 4 + df['fiber'] * 2 +
     df['fruits_veg_nuts'] * 1.5) /
    (df['sugars'] * 3 + df['saturated_fat'] * 2 + df['salt'] * 5 + 1)
)

carbon_75th  = df['carbon_footprint'].quantile(0.75)
is_high_carbon = (df['carbon_footprint'] > carbon_75th).astype(int)
palm_oil_risk  = df['palm_oil_count'] + df['palm_oil_maybe_count'] * 0.5

VITAMIN_REFERENCE = {
    'vitamin_a': 900,  'vitamin_c': 90,   'vitamin_d': 20,
    'vitamin_e': 15,   'vitamin_b1': 1.2, 'vitamin_b2': 1.3,
    'vitamin_b6': 1.7, 'vitamin_b9': 400, 'vitamin_b12': 2.4,
}
vit_score = pd.Series(0.0, index=df.index)
for v, ref in VITAMIN_REFERENCE.items():
    if v in df.columns:
        vit_score += df[v].fillna(0) / ref

mineral_cols_list = [
    'calcium', 'iron', 'magnesium', 'potassium',
    'zinc', 'phosphorus', 'copper', 'manganese', 'selenium'
]
mineral_density = df[mineral_cols_list].fillna(0).sum(axis=1)

derived_cols = pd.DataFrame({
    'processing_score':        processing_score,
    'nova_score':              nova_score,
    'glycemic_index':          glycemic_index,
    'glycemic_load':           glycemic_load,
    'fiber_sugar_ratio':       fiber_sugar_ratio,
    'protein_fat_ratio':       protein_fat_ratio,
    'protein_energy_ratio':    protein_energy_ratio,
    'fat_quality_ratio':       fat_quality_ratio,
    'omega_balance':           omega_balance,
    'carb_fiber_balance':      carb_fiber_balance,
    'energy_density':          energy_density,
    'sodium_salt_ratio':       sodium_salt_ratio,
    'is_high_energy_density':  is_high_energy,
    'sugar_AND_no_fiber':      sugar_AND_no_fiber,
    'high_fat_low_quality':    high_fat_low_quality,
    'additive_AND_processed':  additive_AND_processed,
    'salt_AND_fat_bomb':       salt_AND_fat_bomb,
    'protein_fiber_bonus':     protein_fiber_bonus,
    'sugar_AND_ultra_processed': sugar_AND_ultra_processed,
    'omega3_sat_ratio':        omega3_sat_ratio,
    'whole_food_score':        whole_food_score,
    'macro_balance_score':     macro_balance_score,
    'is_high_carbon':          is_high_carbon,
    'palm_oil_risk':           palm_oil_risk,
    'vitamin_density_score':   vit_score,
    'mineral_density_score':   mineral_density,
}, index=df.index)

# -----------------------------------------------------------------------
# STEP 11 — CONCAT ALL NEW COLUMNS AT ONCE (no fragmentation)
# -----------------------------------------------------------------------
print("\n[7/10] Assembling final dataframe...")

df = pd.concat([
    df,
    pd.DataFrame(cat_cols,     index=df.index),
    pd.DataFrame(subcat_cols,  index=df.index),
    pd.DataFrame(label_cols,   index=df.index),
    pd.DataFrame(allergen_cols,index=df.index),
    pd.DataFrame(additive_cols,index=df.index),
    ingredient_cols.set_index(df.index),
    derived_cols,
], axis=1)

# Drop glycemic_index original (replaced by derived)
# It may be duplicated if already in df — drop one copy
df = df.loc[:, ~df.columns.duplicated()]

print(f"  Assembled dataframe: {df.shape}")

# -----------------------------------------------------------------------
# STEP 12 — FILL NULLS + BUILD TARGETS
# -----------------------------------------------------------------------
print("\n[8/10] Building targets...")

non_fill = ['code', 'product_name', 'nutriscore', 'nutrition_score']
fill_cols = [c for c in df.columns if c not in non_fill]
df[fill_cols] = df[fill_cols].fillna(0)
df['nutriscore'] = df['nutriscore'].fillna('unknown')

# Continuous health score 0-100
df['health_score'] = 100 - ((df['nutrition_score'] + 15) / 55 * 100)
df['health_score'] = df['health_score'].clip(0, 100)

# 3-class health tier
df['health_tier'] = pd.cut(
    df['health_score'],
    bins=[0, 40, 70, 100],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

# -----------------------------------------------------------------------
# STEP 13 — SAMPLE FIRST, THEN FIT TF-IDF ON SAMPLE
# FIX: guarantees tfidf_matrix rows == CSV rows always
# -----------------------------------------------------------------------
print("\n[9/10] Sampling then fitting TF-IDF (aligned)...")

df_sample = df.sample(
    n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

# Pull exact ingredient rows for this sample
sample_ingredients = ingredients_clean.iloc[
    df_sample.index.tolist()].reset_index(drop=True)

tfidf = TfidfVectorizer(
    max_features=300,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
)
tfidf_matrix = tfidf.fit_transform(sample_ingredients.fillna(''))

# Hard assertion — will crash immediately if misaligned
assert tfidf_matrix.shape[0] == len(df_sample), (
    f"SHAPE MISMATCH: tfidf={tfidf_matrix.shape[0]} "
    f"csv={len(df_sample)}"
)
print(f"  TF-IDF matrix : {tfidf_matrix.shape} ✅")
print(f"  CSV rows match: {len(df_sample):,} ✅")

# Save TF-IDF vectorizer
with open(TFIDF_PATH, 'wb') as f:
    pickle.dump(tfidf, f)
print(f"  Saved → {TFIDF_PATH}")

# Save TF-IDF matrix
save_npz(TFIDF_NPZ, tfidf_matrix)
print(f"  Saved → {TFIDF_NPZ}")

# Save CSV
df_sample.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved → {OUTPUT_CSV}")

# -----------------------------------------------------------------------
# STEP 14 — SAVE FEATURE METADATA
# FIX: save with index for correct ordering during inference
# -----------------------------------------------------------------------
print("\n[10/10] Saving feature metadata...")

EXCLUDE_COLS = [
    'code', 'product_name', 'nutriscore',
    'nutrition_score', 'health_score', 'health_tier',
    'expected_salt', 'salt_mismatch', 'data_completeness',
]
feature_cols = [
    c for c in df_sample.columns if c not in EXCLUDE_COLS
]

with open(FEATURES_PATH, 'w') as f:
    for i, col in enumerate(feature_cols):
        f.write(f"{i},{col}\n")
print(f"  Saved {len(feature_cols)} features → {FEATURES_PATH}")

# -----------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"DATA PREPARATION COMPLETE")
print(f"{'='*60}")
print(f"  Rows            : {len(df_sample):,}")
print(f"  Tabular features: {len(feature_cols)}")
print(f"  TF-IDF features : 300")
print(f"  Total features  : {len(feature_cols) + 300}")
print(f"\n  Health score distribution:")
print(df_sample['health_score'].describe().round(2).to_string())
print(f"\n  Health tier distribution:")
for cls, name in [(0,'Unhealthy'),(1,'Moderate'),(2,'Healthy')]:
    count = (df_sample['health_tier'] == cls).sum()
    print(f"    {cls} ({name}): {count:,} ({count/len(df_sample)*100:.1f}%)")
print(f"\n  Nutriscore breakdown:")
print(df_sample['nutriscore'].value_counts().to_string())
print(f"\n  Outputs:")
print(f"    {OUTPUT_CSV}")
print(f"    {TFIDF_PATH}")
print(f"    {TFIDF_NPZ}")
print(f"    {FEATURES_PATH}")
print(f"{'='*60}")
print(f"\nNext step: python ml/health_model.py")