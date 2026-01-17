# Code Documentation

## File Descriptions

### Data Processing

#### `1.py` - Data Preprocessing Pipeline
**Purpose:** Loads raw data, extracts nutrition information, and creates processed datasets

**Key Functions:**
- `parse_nutritions()`: Parses nutrition dict strings from recipe data
- `calculate_bmr()`: Calculates Basal Metabolic Rate using Mifflin-St Jeor equation
- `calculate_target_calories()`: Computes daily calorie target based on user profile

**Input Files:**
- `meal.csv`: Meal definitions
- `recipe.csv`: Recipe data with nutrition info
- `user_meal.csv`: User-meal interactions
- `user_recipe.csv`: User-recipe ratings

**Output Files:**
- `processed_data.pkl`: Binary pickle with all processed data
- `interaction_matrix.csv`: User x Recipe interaction matrix
- `recipe_features.csv`: Recipe features including nutrition
- `user_health_data.csv`: User health profiles

**Key Steps:**
1. Load CSV data
2. Create user-item interaction matrix from ratings (binary: rating >= 4)
3. Parse nutrition information from dictionary strings
4. Generate user health profiles with BMI, goals, conditions
5. Calculate personalized calorie targets
6. Handle missing values with median imputation
7. Save in multiple formats (CSV, pickle) for different uses

---

### Model Training

#### `2_train_ncf.py` - Neural Collaborative Filtering Training
**Purpose:** Trains the NCF model for generating recommendations

**Model Architecture:**
```
Input (User/Item IDs)
    ↓
Embedding Layer (64-dim)
    ↓
Concatenate [User EMB, Item EMB]
    ↓
MLP [128 → 64 → 32 → 1]
    ↓
Sigmoid Output (Interaction Probability)
```

**Key Classes:**
- `InteractionDataset`: PyTorch Dataset for user-item interactions
- `NCF`: Neural Collaborative Filtering model class

**Training Details:**
- **Loss Function:** Binary Cross-Entropy (BCE)
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Learning Rate Schedule:** ReduceLROnPlateau
- **Early Stopping:** Patience=5 epochs
- **Batch Size:** 128
- **Negative Sampling:** 4:1 negative to positive ratio

**Input:** `processed_data.pkl`

**Output Files:**
- `best_ncf_model.pth`: Best model checkpoint
- `user_embeddings.npy`: Learned user embeddings
- `item_embeddings.npy`: Learned recipe embeddings

---

### Model Evaluation

#### `3_evaluate_ncf.py` - NCF Model Evaluation
**Purpose:** Evaluates trained NCF model and generates recommendations

**Key Metrics:**
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: Fraction of relevant recommendations in top-K
- **Recall@K**: Coverage of relevant items in top-K
- **HR@K**: Hit Rate - percentage of users with ≥1 relevant item

**Key Functions:**
- `get_top_k_recommendations()`: Generates top-K recommendations for user

**Output:**
- Top-K recommendations for test users
- Evaluation metrics summary
- `ncf_results.csv`: Evaluation results

---

### Reinforcement Learning

#### `4_rl_environment.py` - RL Environment & Training
**Purpose:** Defines RL environment and trains PPO agent for optimized recommendations

**RL Environment:**
```
State: [User embedding, Current calories, Meal count, Time of day, Nutrient balance]
Action: Select recipe (discrete action space)
Reward: Fitness + Nutrition + Preference - Health penalties
```

**Reward Function Components:**
- **Calorie fit:** Reward based on proximity to target calories
- **Nutrition balance:** Reward for balanced macronutrients
- **Preference:** Reward based on NCF predictions
- **Health penalties:** Penalties for diabetic/hypertension constraints

**Key Classes:**
- `DietRecommendationEnv`: Custom Gym environment

**RL Algorithm:** PPO (Proximal Policy Optimization) from Stable-Baselines3

**Output:**
- `rl_best_model/`: Trained RL agent
- `rl_logs/`: Training logs and evaluations

---

### Model Comparison

#### `5_compare_models.py` - NCF vs NCF+RL Comparison
**Purpose:** Compares recommendation quality between NCF and NCF+RL approaches

**Comparison Metrics:**
- Average calorie deviation
- Average protein per meal
- Average sugar per meal
- Average reward
- User satisfaction proxy

**Output:**
- `comparison_results.csv`: Performance comparison

---

### Research Components

#### `6_final_paper_sections.py` - Paper Results Generation
**Purpose:** Generates research paper sections with results

#### `7_xai_paper_section.py` - Explainable AI Analysis
**Purpose:** Provides explanation components for recommendations

#### `8_paper_checklist.py` - Project Completion Checklist
**Purpose:** Tracks project completion status

#### `final_paper_results.py` - Final Results Summary
**Purpose:** Generates final comprehensive results

---

## Data Flow Diagram

```
Raw Data (CSV files)
    ↓
[1.py] Data Preprocessing
    ↓
Processed Data (processed_data.pkl)
    ├─→ [2_train_ncf.py] NCF Training
    │       ↓
    │   Embeddings (npy) → [4_rl_environment.py] RL Training
    │       ↓                   ↓
    │   best_ncf_model.pth   rl_best_model/
    │       ↓                   ↓
    │   [3_evaluate_ncf.py]    [5_compare_models.py]
    │       ↓                   ↓
    └─→ Results & Metrics → [6_final_paper_sections.py]
                                ↓
                        Research Paper Output
```

---

## Configuration & Hyperparameters

### Model Hyperparameters (in 2_train_ncf.py)
```python
EMBEDDING_DIM = 64          # User/item embedding dimension
MLP_HIDDEN = [128, 64, 32] # MLP layer sizes
LEARNING_RATE = 0.001       # Adam optimizer LR
BATCH_SIZE = 128            # Training batch size
WEIGHT_DECAY = 1e-5         # L2 regularization
DROPOUT = 0.3               # Dropout rate
EPOCHS = 20                 # Max training epochs
PATIENCE = 5                # Early stopping patience
```

### RL Hyperparameters (in 4_rl_environment.py)
```python
EMBEDDING_DIM = 64          # From NCF
CALORIES_LOW = 1500         # Min target calories
CALORIES_HIGH = 3500        # Max target calories
MAX_MEALS_PER_DAY = 3       # Meals in episode
STATE_DIM = 72              # Total state dimensions
ACTION_SPACE = N_RECIPES    # Discrete actions
PPO_TIMESTEPS = 10000       # Total RL training steps
PPO_LR = 3e-4               # PPO learning rate
```

---

## Key Design Decisions

### 1. **Negative Sampling (4:1 Ratio)**
- Accounts for sparsity in interaction matrix
- Prevents model from learning trivial "predict negative" solution
- Balances precision and recall

### 2. **Health-Aware Reward Function**
- Incorporates dietary goals (weight loss, maintenance, muscle gain)
- Considers medical conditions (diabetes, hypertension)
- Balances user preferences with health outcomes

### 3. **Embedding Extraction**
- User/item embeddings from NCF used in RL environment
- Transfers learned representations to RL agent
- Reduces RL training time and improves sample efficiency

### 4. **Binary Classification for NCF**
- Rating ≥ 4 treated as positive (implicit feedback)
- Simplifies from 5-class to 2-class problem
- Improves training stability

### 5. **Separate Train/Val/Test Split**
- 80/10/10 split prevents data leakage
- Validation used for hyperparameter tuning
- Test set for final unbiased evaluation

---

## Common Modifications

### Adjust Model Capacity
```python
# Larger model for more users/items
EMBEDDING_DIM = 128
MLP_HIDDEN = [256, 128, 64, 32]
```

### Faster Training (Lower Quality)
```python
BATCH_SIZE = 256
EMBEDDING_DIM = 32
EPOCHS = 5
```

### Better Quality (Slower Training)
```python
BATCH_SIZE = 64
EMBEDDING_DIM = 128
EPOCHS = 50
PATIENCE = 10
```

### More Health-Conscious
```python
# In RL reward function, increase health weights
reward += 2.0 * health_score  # From 1.0 * health_score
```

---

## Debugging Tips

### Model Not Converging
- Reduce learning rate to 0.0005
- Increase batch size to 256
- Increase embedding dimension

### Out of Memory
- Reduce batch size
- Reduce embedding dimension
- Use gradient checkpointing

### Poor Recommendations
- Increase training epochs
- Increase negative sampling ratio
- Check data quality and ratings distribution

### RL Training Unstable
- Reduce learning rate
- Increase entropy coefficient
- Check reward function scaling

---

## Performance Benchmarks

**Training Time (on typical GPU):**
- Data preprocessing: 30-60s
- NCF training (20 epochs): 5-10 minutes
- RL training (10K steps): 10-15 minutes

**Model Sizes:**
- NCF model: ~2-5 MB
- User embeddings: ~1-5 MB (depending on user count)
- Item embeddings: ~1-10 MB (depending on item count)

**Inference Speed:**
- Recommendations per user: <100ms
- Batch recommendation (100 users): ~5-10s

---

**Last Updated**: January 2026
