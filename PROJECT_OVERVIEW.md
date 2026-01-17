"""
Neural Collaborative Filtering + Reinforcement Learning for Health-Conscious Food Recommendation

Project: NCF22 - Personalized Nutrition Recommendation System
Description:
    This project combines Neural Collaborative Filtering (NCF) with Reinforcement Learning (RL)
    to provide personalized food/recipe recommendations based on user preferences and health profiles.

Components:
    1. Data Preprocessing (1.py)
       - Loads recipe and user interaction data
       - Processes nutrition information
       - Creates user health profiles
       - Generates interaction matrices

    2. NCF Model Training (2_train_ncf.py)
       - Trains Neural Collaborative Filtering model
       - Uses embeddings for users and items
       - Multi-layer perceptron for recommendation

    3. NCF Model Evaluation (3_evaluate_ncf.py)
       - Evaluates NCF model performance
       - Generates recommendations
       - Computes NDCG and other metrics

    4. RL Environment Setup (4_rl_environment.py)
       - Defines RL environment for diet recommendations
       - State: user profile + current meal status
       - Action: select a recipe to recommend
       - Reward: based on nutrition, preferences, and health goals

    5. Model Comparison (5_compare_models.py)
       - Compares NCF vs NCF+RL performance
       - Analyzes recommendation quality

    6. Research Paper Sections (6_final_paper_sections.py)
       - Generates paper sections and results

    7. XAI Analysis (7_xai_paper_section.py)
       - Explainable AI analysis

    8. Project Checklist (8_paper_checklist.py)
       - Tracks project completion

Key Technologies:
    - PyTorch: Neural network framework
    - Gymnasium: RL environment
    - Stable Baselines 3: RL algorithms (PPO)
    - Scikit-learn: ML utilities
    - Pandas/NumPy: Data processing

Author: NCF22 Team
Date: 2024-2026
"""

# Project structure and setup instructions in README.md
