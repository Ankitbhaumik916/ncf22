from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np

# Reuse generator utilities from a valid module name
from meal_plan_utils import (
    load_processed_data,
    try_load_ncf_model,
    score_recipes_for_user,
    build_meal_plan,
)

app = Flask(__name__)
app.secret_key = "ncf22-secret"


def generate_plan_for_user(user_id: str, target_calories: float) -> pd.DataFrame:
    data = load_processed_data()
    if 'recipe_features' not in data or 'user_health_data' not in data:
        raise RuntimeError("Missing data files. Ensure processed_data.pkl or CSVs exist.")

    recipe_features = data['recipe_features']
    user_health_data = data['user_health_data']
    interaction_matrix = data.get('interaction_matrix')

    # Normalize type of user_id
    try:
        if np.issubdtype(user_health_data['user_id'].dtype, np.number):
            user_id = int(user_id)
    except Exception:
        pass

    user_row = user_health_data[user_health_data['user_id'] == user_id]
    if user_row.empty:
        # If not found, return None to display an error
        return None

    is_diabetic = int(user_row.iloc[0].get('is_diabetic', 0))
    default_target = float(user_row.iloc[0].get('target_calories', 2000))
    target_calories = float(target_calories or default_target)

    # Map recipe IDs
    recipe_ids = recipe_features['recipe_id'].tolist()
    num_items = len(recipe_ids)

    # Determine user index
    if interaction_matrix is not None:
        try:
            user_index = list(interaction_matrix.index).index(user_id)
        except ValueError:
            user_index = 0
    else:
        user_index = 0

    model, device = try_load_ncf_model(num_users=len(user_health_data), num_items=num_items)
    ncf_scores = score_recipes_for_user(model, device, user_index, num_items)

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

    return plan_df


@app.get("/")
def index():
    # Preload users for convenience
    data = load_processed_data()
    users = []
    if 'user_health_data' in data:
        df = data['user_health_data']
        preview = df[['user_id', 'is_diabetic', 'target_calories']].head(20)
        users = preview.to_dict('records')
    return render_template("index.html", users=users)


@app.post("/generate")
def generate():
    user_id = request.form.get("user_id", "").strip()
    target_calories = request.form.get("target_calories", "").strip()
    target_calories = float(target_calories) if target_calories else None

    if not user_id:
        flash("Please provide a user_id.", "error")
        return redirect(url_for('index'))

    plan_df = generate_plan_for_user(user_id, target_calories)
    if plan_df is None or plan_df.empty:
        flash("No plan generated. Check user_id and data availability.", "error")
        return redirect(url_for('index'))

    # Convert DataFrame to records for display
    rows = plan_df.to_dict('records')
    return render_template("result.html", user_id=user_id, target_calories=target_calories, rows=rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
