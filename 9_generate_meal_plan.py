import os
import pickle
import math
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def load_processed_data() -> Dict[str, pd.DataFrame]:
    """Load processed_data.pkl if available; else fall back to CSVs."""
    data = {}
    if os.path.exists('processed_data.pkl'):
        with open('processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        # Fallback to raw CSVs expected in workspace root
        if os.path.exists('interaction_matrix.csv'):
            data['interaction_matrix'] = pd.read_csv('interaction_matrix.csv', index_col=0)
        if os.path.exists('recipe_features.csv'):
            data['recipe_features'] = pd.read_csv('recipe_features.csv')
        if os.path.exists('user_health_data.csv'):
            data['user_health_data'] = pd.read_csv('user_health_data.csv')
    return data


def try_load_ncf_model(num_users: int, num_items: int):
    """Load NCF model and weights if available, else return None.
    Defines the NCF class locally to match 2_train_ncf/3_evaluate_ncf.
    """
    try:
        import torch
        import torch.nn as nn

        class NCF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim=64):
                super(NCF, self).__init__()
                self.num_users = num_users
                self.num_items = num_items
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(embedding_dim * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                combined = torch.cat([user_emb, item_emb], dim=-1)
                output = self.mlp(combined)
                return output.squeeze()

        device = 'cuda' if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        model = NCF(num_users, num_items, embedding_dim=64)
        if os.path.exists('best_ncf_model.pth'):
            checkpoint = torch.load('best_ncf_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            return model, device
        return None, None
    except Exception:
        return None, None


def score_recipes_for_user(
    model,
    device,
    user_index: int,
    num_items: int,
) -> np.ndarray:
    """Return per-item scores from NCF for a given user index; zeros if model absent."""
    if model is None:
        return np.zeros(num_items, dtype=np.float32)
    import torch
    with torch.no_grad():
        all_items = torch.arange(num_items).to(device)
        user_idx_tensor = torch.LongTensor([user_index]).to(device)
        user_repeated = user_idx_tensor.repeat(num_items)
        scores = model(user_repeated, all_items).detach().cpu().numpy()
        return scores


def build_meal_plan(
    recipe_df: pd.DataFrame,
    user_health: pd.Series,
    ncf_scores: np.ndarray,
    recipe_ids: List[str],
    target_calories: float,
    is_diabetic: int,
    meal_structure: List[Tuple[str, float]],
) -> pd.DataFrame:
    """Construct a simple daily meal plan selecting recipes matching per-meal calorie targets.

    meal_structure: list of (Meal Name, proportion of daily calories)
    """
    # Ensure necessary columns exist
    required_cols = ['recipe_id', 'recipe_name', 'calories', 'protein', 'carbohydrates', 'sugar', 'fiber']
    for col in required_cols:
        if col not in recipe_df.columns:
            # create default if missing
            recipe_df[col] = 0

    # Prepare working copy
    df = recipe_df.copy()
    df = df[df['recipe_id'].isin(recipe_ids)]
    df = df.dropna(subset=['calories'])

    # Diabetic-friendly filter: cap sugar, prefer fiber, balanced carbs
    sugar_cap = 20.0  # g per serving
    df['diabetic_ok'] = (df['sugar'].fillna(0) <= sugar_cap)

    # Composite suitability score combining NCF and nutrition
    id_to_idx = {rid: i for i, rid in enumerate(recipe_ids)}
    df['ncf_score'] = df['recipe_id'].map(lambda rid: float(ncf_scores[id_to_idx.get(rid, 0)]) if len(ncf_scores) else 0.0)

    # Nutrition score
    def nutrition_score(row):
        score = 0.0
        # Encourage higher protein and fiber
        score += min(row.get('protein', 0) / 30.0, 1.0) * 0.4
        score += min(row.get('fiber', 0) / 10.0, 1.0) * 0.3
        # Penalize sugar for diabetics
        if is_diabetic:
            sugar = row.get('sugar', 0)
            if sugar > 25:
                score -= 0.6
            elif sugar > 15:
                score -= 0.3
        # Light penalty for extremely high carbs
        carbs = row.get('carbohydrates', 0)
        if carbs > 80:
            score -= 0.2
        return score

    df['nutrition_score'] = df.apply(nutrition_score, axis=1)

    # Final suitability score
    df['suitability'] = 0.5 * df['ncf_score'] + 0.5 * df['nutrition_score']

    # Allocate meals
    used_recipe_ids = set()
    rows = []
    for meal_name, proportion in meal_structure:
        meal_target = target_calories * proportion

        # Score closeness to meal_target
        df['calorie_proximity'] = 1.0 - (df['calories'] - meal_target).abs().clip(lower=0, upper=600) / 600.0

        # Prefer diabetic_ok if diabetic
        if is_diabetic:
            candidate_df = df[df['diabetic_ok']]
            if candidate_df.empty:
                candidate_df = df
        else:
            candidate_df = df

        # Exclude already used recipes
        candidate_df = candidate_df[~candidate_df['recipe_id'].isin(used_recipe_ids)]
        if candidate_df.empty:
            candidate_df = df[~df['recipe_id'].isin(used_recipe_ids)]
        if candidate_df.empty:
            break

        # Combined selection score
        candidate_df['selection_score'] = 0.6 * candidate_df['suitability'] + 0.4 * candidate_df['calorie_proximity']
        best = candidate_df.sort_values(['selection_score'], ascending=False).head(1)
        if best.empty:
            continue
        row = best.iloc[0]
        used_recipe_ids.add(row['recipe_id'])

        # Build explanation
        reasons = []
        if row['ncf_score'] > 0.7:
            reasons.append('Similar to past preferences')
        if row['protein'] >= 20:
            reasons.append('High protein')
        if row['sugar'] <= 12:
            reasons.append('Low sugar')
        if row['fiber'] >= 5:
            reasons.append('High fiber')
        reasons.append('Matches meal calorie target')

        rows.append({
            'Meal Time': meal_name,
            'Recipe': row.get('recipe_name', 'Unknown'),
            'Calories': f"{int(round(row['calories']))} kcal",
            'Protein': f"{int(round(row.get('protein', 0)))}g",
            'Carbs': f"{int(round(row.get('carbohydrates', 0)))}g",
            'Sugar': f"{int(round(row.get('sugar', 0)))}g",
            'Fiber': f"{int(round(row.get('fiber', 0)))}g",
            'Why Recommended': ', '.join(reasons)
        })

    plan_df = pd.DataFrame(rows)

    # Append totals
    if not plan_df.empty:
        def parse_int(val):
            try:
                return int(str(val).split()[0])
            except Exception:
                return 0

        total_cal = sum(parse_int(c) for c in plan_df['Calories'])
        total_prot = sum(parse_int(p) for p in plan_df['Protein'])
        total_carbs = sum(parse_int(c) for c in plan_df['Carbs'])
        total_sugar = sum(parse_int(s) for s in plan_df['Sugar'])
        total_fiber = sum(parse_int(f) for f in plan_df['Fiber'])

        plan_df = pd.concat([
            plan_df,
            pd.DataFrame([{
                'Meal Time': 'Total',
                'Recipe': '',
                'Calories': f"{total_cal} kcal",
                'Protein': f"{total_prot}g",
                'Carbs': f"{total_carbs}g",
                'Sugar': f"{total_sugar}g",
                'Fiber': f"{total_fiber}g",
                'Why Recommended': ''
            }])
        ], ignore_index=True)

    return plan_df


def main(user_id: str = None, target_calories: float = None):
    data = load_processed_data()
    if 'recipe_features' not in data or 'user_health_data' not in data:
        print("Missing data files. Ensure processed_data.pkl or CSVs exist.")
        return

    recipe_features = data['recipe_features']
    user_health_data = data['user_health_data']
    interaction_matrix = data.get('interaction_matrix')

    # Determine user
    if user_id is None:
        user_id = user_health_data.iloc[0]['user_id']
    # Normalize type for matching
    try:
        # If user_id column is numeric, cast input
        if np.issubdtype(user_health_data['user_id'].dtype, np.number):
            user_id = int(user_id)
    except Exception:
        pass

    user_row = user_health_data[user_health_data['user_id'] == user_id]
    if user_row.empty:
        print(f"User {user_id} not found; using first user.")
        user_row = user_health_data.iloc[[0]]
        user_id = user_row.iloc[0]['user_id']

    is_diabetic = int(user_row.iloc[0].get('is_diabetic', 0))
    default_target = float(user_row.iloc[0].get('target_calories', 2000))
    target_calories = float(target_calories or default_target)

    # Map IDs
    recipe_ids = recipe_features['recipe_id'].tolist()
    num_items = len(recipe_ids)

    # User index
    if interaction_matrix is not None:
        try:
            user_index = list(interaction_matrix.index).index(user_id)
        except ValueError:
            user_index = 0
    else:
        user_index = 0

    # Try NCF scoring
    model, device = try_load_ncf_model(num_users=len(user_health_data), num_items=num_items)
    ncf_scores = score_recipes_for_user(model, device, user_index, num_items)

    # Define meal structure: Breakfast/Lunch/Dinner/Snack
    meal_structure = [
        ("Breakfast", 0.25),
        ("Lunch", 0.35),
        ("Dinner", 0.30),
        ("Snack", 0.10),
    ]

    plan_df = build_meal_plan(
        recipe_df=recipe_features,
        user_health=user_row.iloc[0],
        ncf_scores=ncf_scores,
        recipe_ids=recipe_ids,
        target_calories=target_calories,
        is_diabetic=is_diabetic,
        meal_structure=meal_structure,
    )

    print(f"\n=== Sample Daily Meal Plan for User {user_id} (Target: {int(target_calories)} kcal) ===")
    if plan_df.empty:
        print("No suitable plan generated. Try relaxing constraints or ensure data is present.")
        return
    # Display nicely
    print(plan_df.to_string(index=False))
    out_file = f"meal_plan_{user_id}.csv"
    plan_df.to_csv(out_file, index=False)
    print(f"\nSaved plan to '{out_file}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a sample daily meal plan")
    parser.add_argument('--user_id', type=str, default=None, help='User ID from user_health_data.csv')
    parser.add_argument('--target_calories', type=float, default=None, help='Daily calorie target')
    args = parser.parse_args()
    main(user_id=args.user_id, target_calories=args.target_calories)
