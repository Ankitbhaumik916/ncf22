"""
Neural Collaborative Filtering (NCF) Model Training Script
==========================================================

This script trains a Neural Collaborative Filtering model for recipe recommendation.

NCF Architecture:
- User embedding layer: maps user IDs to dense user vectors
- Item embedding layer: maps recipe IDs to dense item vectors
- MLP: concatenated embeddings passed through multi-layer perceptron
- Output: sigmoid prediction of user-item interaction probability

Key Features:
- Negative sampling (4:1 ratio) for balanced training
- Train/Val/Test split
- Early stopping based on validation loss
- Learning rate scheduling
- Embedding extraction for RL integration

Author: NCF22 Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset, DataLoader
import time

print("=== LOADING PROCESSED DATA ===")

# Load preprocessed data from Phase 1 (1.py)
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

interaction_matrix = data['interaction_matrix']  # User x Recipe matrix
recipe_features = data['recipe_features']        # Recipe features with nutrition
user_health_data = data['user_health_data']      # User health profiles

print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Recipe features shape: {recipe_features.shape}")
print(f"User health data shape: {user_health_data.shape}")

# === PHASE 1: SAMPLE PREPARATION ===
# Extract positive and negative samples from sparse interaction matrix
print("\n=== PREPARING TRAINING DATA ===")

# Get all positive interactions (feedback = 1, meaning rating >= 4)
positive_interactions = []
user_ids = interaction_matrix.index
recipe_ids = interaction_matrix.columns

for i, user in enumerate(user_ids):
    for j, recipe in enumerate(recipe_ids):
        if interaction_matrix.iloc[i, j] == 1:
            positive_interactions.append((i, j, 1))

print(f"Total positive interactions: {len(positive_interactions)}")

# Generate negative samples (4:1 ratio)
# Negative samples are user-item pairs without recorded interaction
np.random.seed(42)
negative_interactions = []
n_negative = min(len(positive_interactions) * 4, interaction_matrix.size - len(positive_interactions))
sampled_negative = 0

while sampled_negative < n_negative:
    user_idx = np.random.randint(0, len(user_ids))
    recipe_idx = np.random.randint(0, len(recipe_ids))
    
    # Only add if no positive interaction exists (no previous rating)
    if interaction_matrix.iloc[user_idx, recipe_idx] == 0:
        negative_interactions.append((user_idx, recipe_idx, 0))
        sampled_negative += 1

print(f"Negative samples: {len(negative_interactions)}")

# Combine positive and negative samples
all_interactions = positive_interactions + negative_interactions
np.random.shuffle(all_interactions)

print(f"Total samples: {len(all_interactions)}")

# Split into train/val/test
train_data, test_data = train_test_split(all_interactions, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Create PyTorch Dataset
class InteractionDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_idx, recipe_idx, label = self.interactions[idx]
        return torch.LongTensor([user_idx]), torch.LongTensor([recipe_idx]), torch.FloatTensor([label])

train_dataset = InteractionDataset(train_data)
val_dataset = InteractionDataset(val_data)
test_dataset = InteractionDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define NCF Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        # Embedding layers - convert IDs to dense vectors
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Multi-layer perceptron for interaction modeling
        # Input: 2*embedding_dim (concatenated user + item embeddings)
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
            nn.Sigmoid()
        )
        
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass through NCF model.
        Args:
            user_ids: Batch of user IDs
            item_ids: Batch of item IDs
        Returns:
            Predictions: Interaction probability
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        combined = torch.cat([user_emb, item_emb], dim=-1)
        
        # MLP forward pass
        output = self.mlp(combined)
        return output.squeeze()

# Initialize model
num_users = len(user_ids)
num_items = len(recipe_ids)
model = NCF(num_users, num_items, embedding_dim=64)

print(f"\n=== MODEL ARCHITECTURE ===")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for user_batch, item_batch, label_batch in dataloader:
            user_batch = user_batch.to(device).squeeze()
            item_batch = item_batch.to(device).squeeze()
            label_batch = label_batch.to(device).squeeze()
            
            outputs = model(user_batch, item_batch)
            loss = criterion(outputs, label_batch)
            total_loss += loss.item()
            
            # Accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == label_batch).sum().item()
            total += len(label_batch)
    
    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (user_batch, item_batch, label_batch) in enumerate(dataloader):
        user_batch = user_batch.to(device).squeeze()
        item_batch = item_batch.to(device).squeeze()
        label_batch = label_batch.to(device).squeeze()
        
        optimizer.zero_grad()
        outputs = model(user_batch, item_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

# Training loop
print("\n=== TRAINING NCF MODEL ===")
n_epochs = 20
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(n_epochs):
    start_time = time.time()
    
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    print("-" * 50)
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Validate
    val_loss, val_accuracy = evaluate(model, val_loader, device)
    
    epoch_time = time.time() - start_time
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Time: {epoch_time:.2f}s")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, 'best_ncf_model.pth')
        print("  Model saved!")
    else:
        patience_counter += 1
        print(f"  No improvement for {patience_counter} epochs")
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
print("\n=== LOADING BEST MODEL ===")
checkpoint = torch.load('best_ncf_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Final evaluation on test set
print("\n=== FINAL TEST EVALUATION ===")
test_loss, test_accuracy = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Recommendation function (Top-K)
def get_top_k_recommendations(model, user_idx, k=10, device='cpu'):
    """Get top-K recommendations for a user"""
    model.eval()
    user_idx_tensor = torch.LongTensor([user_idx]).to(device)
    
    # Score all items
    with torch.no_grad():
        all_items = torch.arange(num_items).to(device)
        user_repeated = user_idx_tensor.repeat(num_items)
        scores = model(user_repeated, all_items)
    
    # Get top-K
    top_k_scores, top_k_indices = torch.topk(scores, k)
    
    return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()

# Test recommendation for a sample user
print("\n=== SAMPLE RECOMMENDATIONS ===")
sample_user_idx = 0  # First user
top_k_indices, top_k_scores = get_top_k_recommendations(model, sample_user_idx, k=5, device=device)

print(f"Top 5 recommendations for user {user_ids[sample_user_idx]}:")
for i, (recipe_idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
    recipe_id = recipe_ids[recipe_idx]
    recipe_name = recipe_features[recipe_features['recipe_id'] == recipe_id]['recipe_name'].values
    if len(recipe_name) > 0:
        recipe_name = recipe_name[0][:50] + "..." if len(recipe_name[0]) > 50 else recipe_name[0]
    else:
        recipe_name = "Unknown"
    
    print(f"  {i+1}. Recipe ID: {recipe_id}, Score: {score:.4f}")
    print(f"     Name: {recipe_name}")

# Save model embeddings for RL
print("\n=== SAVING EMBEDDINGS FOR RL ===")
user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

np.save('user_embeddings.npy', user_embeddings)
np.save('item_embeddings.npy', item_embeddings)

print(f"User embeddings shape: {user_embeddings.shape}")
print(f"Item embeddings shape: {item_embeddings.shape}")

print("\n✅ NCF TRAINING COMPLETE!")