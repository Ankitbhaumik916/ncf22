"""
Data Preprocessing Script for Health-Conscious Food Recommendation System
=========================================================================

This script performs data preprocessing and feature engineering for the NCF+RL
recommendation system. It:
1. Loads raw data (meals, recipes, user interactions)
2. Processes nutrition information from recipes
3. Creates user health profiles
4. Generates interaction matrices
5. Saves processed data for model training

Author: NCF22 Team
Date: 2024
"""

import pandas as pd
import numpy as np
import ast  # For parsing Python dictionaries
import pickle

# Load data
meals = pd.read_csv('meal.csv')
recipes = pd.read_csv('recipe.csv')
user_meals = pd.read_csv('user_meal.csv')
user_recipes = pd.read_csv('user_recipe.csv')

print("=== DATASET INFO ===")
print(f"Meals (bundles): {meals.shape}")
print(f"Recipes (individual): {recipes.shape}")
print(f"User-Meal interactions: {user_meals.shape}")
print(f"User-Recipe ratings: {user_recipes.shape}")

# === PHASE 1: PREPARE INTERACTION DATA ===
# We use user_recipes for NCF as it provides granular user-item interactions
# with explicit ratings (more signal than implicit meal-level interactions)
print(f"Unique users in user_recipes: {user_recipes['user_id'].nunique()}")
print(f"Unique recipes in user_recipes: {user_recipes['recipe_id'].nunique()}")
print(f"Rating range: {user_recipes['rating'].min()} to {user_recipes['rating'].max()}")

# Create implicit feedback (rating >= 4 = positive)
user_recipes['feedback'] = (user_recipes['rating'] >= 4).astype(int)
print(f"\nPositive feedback (rating >= 4): {user_recipes['feedback'].sum()} / {len(user_recipes)}")

# Create user-item matrix for NCF
interaction_matrix = user_recipes.pivot_table(
    index='user_id',
    columns='recipe_id',
    values='feedback',
    fill_value=0
)

print(f"\nInteraction matrix shape: {interaction_matrix.shape}")
print(f"Sparsity: {(interaction_matrix == 0).sum().sum() / interaction_matrix.size:.2%}")

# Get recipe nutrition info (we'll parse the nutritions column)
print("\n=== NUTRITION DATA EXAMPLE ===")
print(recipes['nutritions'].iloc[0])
print(f"Type: {type(recipes['nutritions'].iloc[0])}")

# === PHASE 2: EXTRACT NUTRITION FEATURES ===
# Parse nutrition data from dictionary strings in recipe data
def parse_nutritions(nutrition_str):
    """
    Parse nutrition information from Python dictionary string format.
    
    Args:
        nutrition_str: String representation of nutrition dict
        
    Returns:
        dict: Parsed nutrition values (calories, protein, fat, etc.)
    """
    nutrients = {}
    try:
        # Convert string to dictionary using ast.literal_eval
        if isinstance(nutrition_str, str):
            # Remove 'u' prefix for Python 3 compatibility
            nutrition_str = nutrition_str.replace("u'", "'").replace('u"', '"')
            nut_dict = ast.literal_eval(nutrition_str)
            
            # Extract key nutrient values
            key_mapping = {
                'calories': ['calories', 'calories'],
                'protein': ['protein', 'protein'],
                'fat': ['fat', 'fat'],
                'carbohydrates': ['carbohydrates', 'carbs'],
                'sugar': ['sugars', 'sugar'],
                'sodium': ['sodium', 'salt'],
                'fiber': ['fiber', 'dietary fiber'],
                'cholesterol': ['cholesterol', 'cholesterol']
            }
            
            for key, possible_names in key_mapping.items():
                for name in possible_names:
                    if name in nut_dict:
                        nutrients[key] = nut_dict[name].get('amount', np.nan)
                        break
    except Exception as e:
        # If parsing fails, return empty dict
        pass
    return nutrients

# Test parsing
test_nutrition = recipes['nutritions'].iloc[0]
print(f"\nParsing example: {test_nutrition[:100]}...")
parsed = parse_nutritions(test_nutrition)
print(f"Parsed nutrients: {parsed}")

# Apply to all recipes
print("\nParsing nutrition data for all recipes...")
recipes['parsed_nutritions'] = recipes['nutritions'].apply(parse_nutritions)

# Extract common nutrients
nutrient_columns = ['calories', 'protein', 'fat', 'carbohydrates', 'sugar', 'sodium', 'fiber', 'cholesterol']
for nutrient in nutrient_columns:
    recipes[nutrient] = recipes['parsed_nutritions'].apply(
        lambda x: x.get(nutrient, np.nan)
    )

print(f"\nNutrition columns extracted - Sample:")
print(recipes[['recipe_id', 'recipe_name'] + nutrient_columns].head())

# Check for missing values
print(f"\n=== MISSING VALUE COUNT ===")
for col in nutrient_columns:
    missing = recipes[col].isna().sum()
    total = len(recipes)
    print(f"{col}: {missing}/{total} missing ({missing/total:.1%})")

# Fill missing values with median (only for numeric columns)
numeric_cols = [col for col in nutrient_columns if recipes[col].dtype in [np.float64, np.int64]]
for col in numeric_cols:
    recipes[col] = recipes[col].fillna(recipes[col].median())

print(f"\nAfter filling: {recipes[numeric_cols].isna().sum().sum()} missing values remaining")

# Create simulated health data for users
user_ids = interaction_matrix.index
n_users = len(user_ids)

# Simulate realistic user profiles
np.random.seed(42)  # For reproducibility

user_health_data = pd.DataFrame({
    'user_id': user_ids,
    'age': np.random.randint(20, 65, n_users),
    'weight_kg': np.random.uniform(50, 120, n_users),
    'height_cm': np.random.uniform(150, 190, n_users),
    'bmi': np.random.uniform(18.5, 35, n_users),
    'daily_steps': np.random.randint(2000, 15000, n_users),
    'activity_level': np.random.choice([1, 2, 3, 4, 5], n_users, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    'is_diabetic': np.random.choice([0, 1], n_users, p=[0.85, 0.15]),
    'has_hypertension': np.random.choice([0, 1], n_users, p=[0.8, 0.2]),
    'dietary_goal': np.random.choice(['weight_loss', 'maintenance', 'muscle_gain'], n_users)
})

# Calculate BMR and target calories
def calculate_bmr(row):
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation.
    
    BMR represents calories burned at rest.
    
    Args:
        row: DataFrame row with age, weight_kg, height_cm
        
    Returns:
        float: BMR in calories/day
    """
    # Mifflin-St Jeor Equation
    if row['age'] < 18:
        return 0
    bmr = 10 * row['weight_kg'] + 6.25 * row['height_cm'] - 5 * row['age']
    bmr += 5  # Add constant (simplified)
    return bmr

def calculate_target_calories(row):
    """
    Calculate daily target calorie intake based on BMR and lifestyle.
    
    Applies activity multipliers (TDEE) and adjusts for dietary goals.
    
    Args:
        row: DataFrame row with BMR, activity_level, dietary_goal
        
    Returns:
        float: Daily calorie target
    """
    bmr = calculate_bmr(row)
    # Activity multipliers (sedentary to very active)
    multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    tdee = bmr * multipliers.get(row['activity_level'], 1.55)
    
    # Adjust for dietary goal (weight loss = -15%, muscle gain = +15%)
    if row['dietary_goal'] == 'weight_loss':
        return tdee * 0.85
    elif row['dietary_goal'] == 'muscle_gain':
        return tdee * 1.15
    else:
        return tdee

user_health_data['bmr'] = user_health_data.apply(calculate_bmr, axis=1)
user_health_data['target_calories'] = user_health_data.apply(calculate_target_calories, axis=1)

print("\n=== USER HEALTH DATA SAMPLE ===")
print(user_health_data.head())

# Save processed data
print("\n=== SAVING PROCESSED DATA ===")

# 1. Interaction matrix for NCF
interaction_matrix.to_csv('interaction_matrix.csv')
print(f"Saved interaction matrix: {interaction_matrix.shape}")

# 2. Recipe features (with nutrition)
# Only select numeric columns for median filling
recipe_features = recipes[['recipe_id', 'recipe_name', 'category', 'aver_rate']].copy()
for col in numeric_cols:
    recipe_features[col] = recipes[col]

# Fill any remaining NaN in aver_rate (rating)
recipe_features['aver_rate'] = recipe_features['aver_rate'].fillna(recipe_features['aver_rate'].median())

recipe_features.to_csv('recipe_features.csv', index=False)
print(f"Saved recipe features: {recipe_features.shape}")

# 3. User health data
user_health_data.to_csv('user_health_data.csv', index=False)
print(f"Saved user health data: {user_health_data.shape}")

# 4. Save for quick reload
# Binary pickle format for fast loading of large data structures
data_dict = {
    'interaction_matrix': interaction_matrix,
    'recipe_features': recipe_features,
    'user_health_data': user_health_data,
    'full_recipes': recipes  # Full recipes for reference
}

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print("\n✅ DATA PROCESSING COMPLETE!")
print(f"- Users: {len(user_ids)}")
print(f"- Recipes: {recipe_features.shape[0]}")
print(f"- Interactions: {interaction_matrix.sum().sum()}")
print(f"- Sparsity: {(interaction_matrix == 0).sum().sum() / interaction_matrix.size:.2%}")
print(f"- Nutrition features extracted: {numeric_cols}")

# Show statistics
print("\n=== RECIPE NUTRITION STATISTICS ===")
for col in numeric_cols[:6]:  # Show first 6 nutrients
    if col in recipe_features.columns:
        print(f"{col}: mean={recipe_features[col].mean():.1f}, min={recipe_features[col].min():.1f}, max={recipe_features[col].max():.1f}")