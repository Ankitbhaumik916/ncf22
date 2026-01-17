# Health-Conscious Food Recommendation System (NCF22)

## Project Overview

This project combines **Neural Collaborative Filtering (NCF)** with **Reinforcement Learning (RL)** to create a personalized food recommendation system that considers both user preferences and health profiles.

### Key Features

- **Neural Collaborative Filtering**: Uses embeddings to learn user-item interactions and generate recommendations
- **Health-Aware**: Incorporates user health profiles (BMI, dietary goals, medical conditions)
- **Reinforcement Learning**: Optimizes recommendations based on nutritional constraints and user preferences
- **Nutrition Analysis**: Extracts and analyzes recipe nutrition information
- **Explainable AI**: Provides interpretable recommendations

## Project Structure

```
ncf22/
├── 1.py                          # Data preprocessing and feature engineering
├── 2_train_ncf.py               # NCF model training
├── 3_evaluate_ncf.py            # NCF model evaluation
├── 4_rl_environment.py          # RL environment definition
├── 5_compare_models.py          # Compare NCF vs NCF+RL
├── 6_final_paper_sections.py    # Generate paper results
├── 7_xai_paper_section.py       # Explainable AI analysis
├── 8_paper_checklist.py         # Project completion checklist
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ncf22
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Data Preprocessing
Process raw data and create interaction matrices:
```bash
python 1.py
```

Output: `processed_data.pkl`, `interaction_matrix.csv`, `recipe_features.csv`, `user_health_data.csv`

### Step 2: Train NCF Model
Train the Neural Collaborative Filtering model:
```bash
python 2_train_ncf.py
```

Output: `best_ncf_model.pth`, `user_embeddings.npy`, `item_embeddings.npy`

### Step 3: Evaluate NCF
Evaluate the NCF model and generate recommendations:
```bash
python 3_evaluate_ncf.py
```

Output: `ncf_results.csv`

### Step 4: Setup RL Environment
Create RL environment for optimized recommendations:
```bash
python 4_rl_environment.py
```

### Step 5: Compare Models
Compare NCF vs NCF+RL performance:
```bash
python 5_compare_models.py
```

Output: `comparison_results.csv`

## Dataset Requirements

The system expects the following CSV files:
- `meal.csv`: Meal/bundle definitions
- `recipe.csv`: Individual recipe data with nutrition information
- `user_meal.csv`: User-meal interactions
- `user_recipe.csv`: User-recipe ratings

### Expected Schema

**recipe.csv**:
- `recipe_id`: Unique recipe identifier
- `recipe_name`: Recipe name
- `category`: Recipe category
- `nutritions`: Nutrition info (dict format)
- `aver_rate`: Average rating

**user_recipe.csv**:
- `user_id`: Unique user identifier
- `recipe_id`: Recipe identifier
- `rating`: User rating (typically 1-5)

**user_health_data.csv** (generated):
- `user_id`: User identifier
- `age`: User age
- `bmi`: Body Mass Index
- `dietary_goal`: 'weight_loss', 'maintenance', or 'muscle_gain'
- `is_diabetic`: Binary flag
- `has_hypertension`: Binary flag
- `target_calories`: Daily calorie target

## Model Architecture

### Neural Collaborative Filtering

```
User Input (user_id) --> User Embedding (64-dim)
                              |
                              +---> MLP Layers --> Output (1)
                              |
Item Input (recipe_id) --> Item Embedding (64-dim)

MLP: [128, 64, 32, 1] with ReLU and Dropout
```

### Reward Function (RL)

```
Reward = calorie_fit + nutrition_balance + preference_score - health_penalties
```

## Results

The system produces:
- **NCF Metrics**: NDCG, Precision@K, Recall@K, HR@K
- **RL Metrics**: Average reward, calorie accuracy, user satisfaction proxy
- **Comparison**: Performance delta between NCF and NCF+RL

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dimension | 64 | Size of user/item embeddings |
| MLP Hidden Sizes | [128, 64, 32] | MLP layer dimensions |
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Batch Size | 128 | Training batch size |
| Negative Ratio | 4:1 | Negative:positive sampling ratio |
| RL Algorithm | PPO | Proximal Policy Optimization |
| RL Steps | 10,000 | Total RL training steps |

## Dependencies

Key libraries:
- `torch`: Deep learning framework
- `gymnasium`: RL environment toolkit
- `stable-baselines3`: RL algorithms
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: ML utilities
- `scipy`: Scientific computing

See `requirements.txt` for complete list.

## Technical Details

### Data Pipeline
1. **Load**: Read meal, recipe, and interaction CSVs
2. **Parse**: Extract nutrition information from dictionaries
3. **Engineer**: Create features (BMI, calorie targets, nutrient balances)
4. **Normalize**: Scale features to [-1, 1]
5. **Split**: 80% train, 20% test with stratification

### Training Process
- **Positive Sampling**: Interactions with rating ≥ 4
- **Negative Sampling**: Random items without positive interaction (4:1 ratio)
- **Loss Function**: Binary cross-entropy with BCE logits
- **Optimization**: Adam with lr=0.001
- **Regularization**: L2 weight decay (0.0001), Dropout (0.3)

### Evaluation Metrics
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: Fraction of top-K recommendations that are relevant
- **Recall@K**: Fraction of relevant items in top-K
- **Hit Rate@K**: Percentage of users with ≥1 relevant item in top-K

## Advanced Features

### Health-Aware Recommendation
Recommendations consider:
- User's dietary goals (weight loss, maintenance, muscle gain)
- Medical conditions (diabetes, hypertension)
- Calorie budget and nutritional targets
- Macro and micronutrient balance

### Explainability
The system can explain recommendations through:
- Embedding space visualization
- Relevant user/item features
- Contribution scores from MLP layers
- RL decision reasoning

## Performance Considerations

- **Sparsity**: Typical interaction matrix sparsity ~95-98%
- **Scalability**: Can handle 1000s of users and 10000s of items
- **Inference**: Recommendation generation in <100ms per user
- **Training**: Full model training in ~5-10 minutes (GPU recommended)

## Troubleshooting

### Common Issues

**Memory Error**: Reduce batch size or embedding dimension
```python
# In 2_train_ncf.py
BATCH_SIZE = 64  # Reduce from 128
EMBEDDING_DIM = 32  # Reduce from 64
```

**Slow Training**: Enable GPU
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Poor Recommendations**: Increase training steps or embedding dimension
```python
EPOCHS = 50  # Increase training
EMBEDDING_DIM = 128  # Larger embeddings
```

## Future Enhancements

- [ ] Side information fusion (recipe ingredients, user location)
- [ ] Temporal dynamics (seasonal recipes, user preferences over time)
- [ ] Multi-stakeholder optimization (taste vs. health tradeoff)
- [ ] Cold-start solutions for new users/items
- [ ] Real-time feedback integration
- [ ] Mobile app integration

## Citation

If you use this project in your research, please cite:

```bibtex
@project{ncf22,
  title={Health-Conscious Food Recommendation using NCF and RL},
  author={NCF22 Team},
  year={2024}
}
```

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Last Updated**: January 2026
**Project Status**: Active Development
