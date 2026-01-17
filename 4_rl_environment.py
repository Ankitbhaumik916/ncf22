import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import warnings
warnings.filterwarnings('ignore')

print("=== SETTING UP RL ENVIRONMENT ===")

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

interaction_matrix = data['interaction_matrix']
recipe_features = data['recipe_features']
user_health_data = data['user_health_data']

# Load NCF embeddings
user_embeddings = np.load('user_embeddings.npy')
item_embeddings = np.load('item_embeddings.npy')

print(f"User embeddings shape: {user_embeddings.shape}")
print(f"Item embeddings shape: {item_embeddings.shape}")

# Create recipe dictionary for quick lookup
recipe_dict = recipe_features.set_index('recipe_id').to_dict('index')

# Create user dictionary
user_dict = user_health_data.set_index('user_id').to_dict('index')

print(f"\nTotal users: {len(user_dict)}")
print(f"Total recipes: {len(recipe_dict)}")

# Define RL Environment
class DietRecommendationEnv(gym.Env):
    """
    RL Environment for Diet Recommendation
    State: User profile + current meal status
    Action: Recommend a recipe
    Reward: Based on nutrition, preference, and health goals
    """
    
    def __init__(self, user_id, max_meals_per_day=3):
        super(DietRecommendationEnv, self).__init__()
        
        self.user_id = user_id
        self.user_idx = list(interaction_matrix.index).index(user_id) if user_id in interaction_matrix.index else 0
        self.max_meals = max_meals_per_day
        
        # Get user info
        user_info = user_dict.get(user_id, {})
        self.target_calories = user_info.get('target_calories', 2000)
        self.bmi = user_info.get('bmi', 25)
        self.is_diabetic = user_info.get('is_diabetic', 0)
        self.has_hypertension = user_info.get('has_hypertension', 0)
        self.dietary_goal = user_info.get('dietary_goal', 'maintenance')
        
        # State space: [user_embedding, current_calories, meal_count, time_of_day, nutrient_balance]
        state_dim = user_embeddings.shape[1] + 4 + 4  # user_emb + 4 meal features + 4 nutrient balances
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_dim,), dtype=np.float32)
        
        # Action space: Select from available recipes
        self.action_space = spaces.Discrete(len(recipe_dict))
        
        # Available recipes (simplified - all recipes)
        self.recipe_ids = list(recipe_dict.keys())
        self.recipe_indices = {recipe_id: idx for idx, recipe_id in enumerate(self.recipe_ids)}
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment for a new day"""
        super().reset(seed=seed)
        self.current_calories = 0
        self.meal_count = 0
        self.time_of_day = 0  # 0=breakfast, 1=lunch, 2=dinner
        self.nutrient_balance = np.zeros(4)  # [protein, carbs, fat, fiber] balance
        
        # Get user embedding
        self.user_embedding = user_embeddings[self.user_idx]
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state vector"""
        # Normalize features
        calorie_ratio = self.current_calories / self.target_calories if self.target_calories > 0 else 0
        meal_ratio = self.meal_count / self.max_meals
        time_norm = self.time_of_day / 3.0  # 0-1 scale
        
        # Create state vector
        state = np.concatenate([
            self.user_embedding,  # User preferences
            [calorie_ratio, meal_ratio, time_norm, self.bmi / 40.0],  # Meal context
            self.nutrient_balance  # Nutrient balance
        ])
        
        # Normalize to [-1, 1]
        state = np.clip(state, -1, 1)
        
        return state.astype(np.float32)
    
    def _calculate_meal_suitability(self, recipe_id):
        """Calculate how suitable a recipe is for current user"""
        recipe_info = recipe_dict.get(recipe_id, {})
        
        # 1. Preference score from NCF
        recipe_idx = self.recipe_indices.get(recipe_id, 0)
        preference_score = 0.5  # Default
        
        # 2. Nutrition score
        calories = recipe_info.get('calories', 0)
        protein = recipe_info.get('protein', 0)
        carbs = recipe_info.get('carbohydrates', 0)
        fat = recipe_info.get('fat', 0)
        sugar = recipe_info.get('sugar', 0)
        sodium = recipe_info.get('sodium', 0)
        fiber = recipe_info.get('fiber', 0)
        
        nutrition_score = 0
        
        # Protein check (good)
        if protein > 10:
            nutrition_score += 0.2
        
        # Sugar check (bad for diabetics)
        if self.is_diabetic and sugar > 15:
            nutrition_score -= 0.3
        
        # Sodium check (bad for hypertension)
        if self.has_hypertension and sodium > 500:
            nutrition_score -= 0.3
        
        # Fiber check (good)
        if fiber > 5:
            nutrition_score += 0.1
        
        # 3. Calorie appropriateness
        remaining_calories = self.target_calories - self.current_calories
        meals_remaining = self.max_meals - self.meal_count
        ideal_calories_per_meal = remaining_calories / max(1, meals_remaining)
        
        calorie_diff = abs(calories - ideal_calories_per_meal)
        calorie_score = 1.0 - min(calorie_diff / 500.0, 1.0)
        
        # Combine scores
        total_score = 0.3 * preference_score + 0.4 * nutrition_score + 0.3 * calorie_score
        
        return total_score, calories, protein, carbs, fat
    
    def step(self, action):
        """
        Take action (recommend recipe)
        Returns: next_state, reward, done, info
        """
        # Get recipe ID from action
        recipe_id = self.recipe_ids[action]
        
        # Calculate meal suitability
        suitability_score, calories, protein, carbs, fat = self._calculate_meal_suitability(recipe_id)
        
        # Update state
        self.current_calories += calories
        self.meal_count += 1
        self.time_of_day = min(self.time_of_day + 1, 2)
        
        # Update nutrient balance
        self.nutrient_balance[0] += protein / 50.0  # Normalize protein
        self.nutrient_balance[1] += carbs / 100.0   # Normalize carbs
        self.nutrient_balance[2] += fat / 50.0      # Normalize fat
        self.nutrient_balance[3] += 0.1             # Simple fiber tracking
        
        # Calculate reward
        reward = self._calculate_reward(suitability_score, calories)
        
        # Check if day is done
        done = self.meal_count >= self.max_meals or self.current_calories >= self.target_calories
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'recipe_id': recipe_id,
            'calories': calories,
            'total_calories': self.current_calories,
            'meal_count': self.meal_count,
            'suitability_score': suitability_score
        }
        
        return next_state, reward, done, False, info
    
    def _calculate_reward(self, suitability_score, meal_calories):
        """Calculate reward for the action - IMPROVED"""
        # Base reward from suitability
        reward = suitability_score
        
        # STRONGER calorie targeting (weight increased)
        remaining_calories = self.target_calories - self.current_calories
        meals_remaining = self.max_meals - self.meal_count
        
        if meals_remaining > 0:
            ideal_per_meal = remaining_calories / meals_remaining
            # Stronger penalty for deviation - encourage hitting calorie targets
            calorie_diff = abs(meal_calories - ideal_per_meal)
            calorie_penalty = min(calorie_diff / 300.0, 1.0)  # More sensitive, capped at 1
            reward -= 0.5 * calorie_penalty  # Increased from 0.2 to 0.5
        
        # End-of-day calorie target reward (strongly encourage hitting target)
        if self.meal_count >= self.max_meals:
            final_diff = abs(self.current_calories - self.target_calories)
            if final_diff < 200:  # Within 200 calories
                reward += 1.0
            elif final_diff < 400:
                reward += 0.5
            else:
                reward -= 0.5 * (final_diff / 1000.0)  # Penalty for missing target
        
        # Meal timing bonus (encourage regular meals)
        if self.meal_count <= self.max_meals:
            reward += 0.1
        
        # STRONGER health constraint penalties
        recipe_info = recipe_dict.get(self.recipe_ids[0], {})
        sugar = recipe_info.get('sugar', 0)
        sodium = recipe_info.get('sodium', 0)
        
        # More aggressive penalties for diabetics
        if self.is_diabetic:
            if sugar > 20:
                reward -= 0.8  # Increased from 0.3
            elif sugar > 15:
                reward -= 0.4
        
        # More aggressive penalties for hypertension
        if self.has_hypertension:
            if sodium > 1000:
                reward -= 0.8  # Increased from 0.3
            elif sodium > 700:
                reward -= 0.4
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"User: {self.user_id}")
        print(f"Meals: {self.meal_count}/{self.max_meals}")
        print(f"Calories: {self.current_calories:.0f}/{self.target_calories:.0f}")
        print(f"Time: {'Breakfast' if self.time_of_day == 0 else 'Lunch' if self.time_of_day == 1 else 'Dinner'}")
        print(f"Nutrient Balance: {self.nutrient_balance}")

# Test environment with a sample user
print("\n=== TESTING ENVIRONMENT ===")
sample_user_id = user_health_data.iloc[0]['user_id']
env = DietRecommendationEnv(sample_user_id, max_meals_per_day=3)

# Test reset
state, info = env.reset()
print(f"State shape: {state.shape}")
print(f"State sample: {state[:5]}...")

# Test a random action
action = env.action_space.sample()
next_state, reward, done, truncated, info = env.step(action)

print(f"\nTest Action Results:")
print(f"  Action (recipe index): {action}")
print(f"  Recipe ID: {info['recipe_id']}")
print(f"  Calories added: {info['calories']:.1f}")
print(f"  Reward: {reward:.3f}")
print(f"  Done: {done}")
print(f"  Next state shape: {next_state.shape}")

# Create a vectorized environment for training
print("\n=== CREATING VECTORIZED ENVIRONMENT ===")

def make_env(user_id):
    def _init():
        return DietRecommendationEnv(user_id)
    return _init

# Select a subset of users for training (for speed)
train_users = user_health_data['user_id'].iloc[:50].tolist()  # First 50 users
print(f"Training on {len(train_users)} users")

# Create vectorized environment
env = DummyVecEnv([make_env(user_id) for user_id in train_users])

# Initialize PPO model
print("\n=== INITIALIZING PPO MODEL ===")
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

print("Model architecture:")
print(model.policy)

# Training callback
print("\n=== SETTING UP TRAINING ===")
eval_env = DummyVecEnv([make_env(train_users[0])])  # Use first user for evaluation

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./rl_best_model/',
    log_path='./rl_logs/',
    eval_freq=1000,
    deterministic=True,
    render=False,
    verbose=1
)

# Train the model
print("\n=== TRAINING RL AGENT ===")
print("This will take some time...")
print("Training for 50,000 timesteps (~5-10 minutes)")

try:
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        progress_bar=False  # Disabled to avoid tqdm dependency
    )
    
    # Save the trained model
    model.save("diet_rl_agent")
    print("\n✅ RL AGENT TRAINING COMPLETE!")
    print("Model saved as 'diet_rl_agent.zip'")
    
except Exception as e:
    print(f"\n⚠️ Training error: {e}")
    print("Saving partial model...")
    model.save("diet_rl_agent_partial")
    print("Partial model saved.")

# Test the trained agent
print("\n=== TESTING TRAINED AGENT ===")
test_user_id = train_users[0]
test_env = DietRecommendationEnv(test_user_id)

# Reset environment
state, info = test_env.reset()
done = False
total_reward = 0
meal_plan = []

while not done:
    # Get action from trained model
    action, _states = model.predict(state, deterministic=True)
    
    # Take action
    state, reward, done, truncated, info = test_env.step(action)
    total_reward += reward
    meal_plan.append(info['recipe_id'])
    
    if done or truncated:
        break

print(f"\nTest User: {test_user_id}")
print(f"Target Calories: {test_env.target_calories:.0f}")
print(f"Total Calories Consumed: {test_env.current_calories:.0f}")
print(f"Total Reward: {total_reward:.3f}")
print(f"Number of Meals: {len(meal_plan)}")
print(f"Meal Plan Recipe IDs: {meal_plan}")

# Show details of recommended meals
print("\n=== MEAL PLAN DETAILS ===")
for i, recipe_id in enumerate(meal_plan):
    recipe_info = recipe_dict.get(recipe_id, {})
    print(f"\nMeal {i+1} (Recipe ID: {recipe_id}):")
    print(f"  Name: {recipe_info.get('recipe_name', 'Unknown')[:50]}...")
    print(f"  Calories: {recipe_info.get('calories', 0):.1f}")
    print(f"  Protein: {recipe_info.get('protein', 0):.1f}g")
    print(f"  Carbs: {recipe_info.get('carbohydrates', 0):.1f}g")
    print(f"  Sugar: {recipe_info.get('sugar', 0):.1f}g")

print("\n✅ RL IMPLEMENTATION COMPLETE!")
print("\nNext: Run evaluation and compare NCF vs NCF+RL")