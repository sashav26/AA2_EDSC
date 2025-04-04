import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from dataset_loader import BaggageDataset
from model import FlightBaggageModel

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "final_training_dataset.parquet"
best_model_path = "./models_run2/model_AdamW_lr0.0005_hs128-64_dr0.2_wd0.001_bs64_schReduceLROnPlateau.pt"  # Update as needed

# -------------------------------
# Load Best Model
# -------------------------------
# Assume best model uses these parameters (adjust as needed)
hidden_sizes = [128, 64]
model = FlightBaggageModel(num_aircraft_types=10, num_dayofweek=7,
                           embed_dim_aircraft=4, embed_dim_day=3,
                           hidden_sizes=hidden_sizes,
                           dropout_rate=0.2).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Loaded best model from {best_model_path}")

# -------------------------------
# Load Dataset and Create Test Split
# -------------------------------
dataset = BaggageDataset(dataset_path)
dataset_size = len(dataset)
print(f"Total flights in dataset: {dataset_size}")

# Best practice: test on unseen data. We use the same split as before:
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
_, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test set size: {len(test_dataset)}")

# -------------------------------
# Define Loss Functions for Metrics
# -------------------------------
criterion_dist = nn.KLDivLoss(reduction="batchmean")
criterion_count = nn.MSELoss()

# -------------------------------
# Evaluate the Model on Test Set and Collect Metrics
# -------------------------------
model.eval()
total_kl_loss = 0.0
total_mse_loss = 0.0
total_mae_loss = 0.0
n = 0

# Also store errors per flight for visualization
count_errors = []
for batch in test_loader:
    input_vec, target_dist, target_count = batch
    aircraft_type = torch.zeros(input_vec.size(0), dtype=torch.long, device=device)
    day_of_week = input_vec[:, 4].long()  # day_of_week is at index 4
    continuous_feats = torch.stack([input_vec[:, 1], input_vec[:, 0],
                                      input_vec[:, 3], input_vec[:, 2]], dim=1).to(device)
    target_dist = target_dist.to(device)
    target_count = target_count.to(device)
    
    with torch.no_grad():
        pred_dist, count_pred = model(aircraft_type, day_of_week, continuous_feats)
    
    # Compute losses
    kl_loss = criterion_dist(torch.log(pred_dist + 1e-8), target_dist)
    mse_loss = criterion_count(count_pred.squeeze(), target_count)
    mae_loss = torch.mean(torch.abs(count_pred.squeeze() - target_count))
    
    batch_size = input_vec.size(0)
    total_kl_loss += kl_loss.item() * batch_size
    total_mse_loss += mse_loss.item() * batch_size
    total_mae_loss += mae_loss.item() * batch_size
    n += batch_size
    
    # For each flight in the batch, store the absolute count error
    count_errors.extend(torch.abs(count_pred.squeeze() - target_count).cpu().numpy().tolist())

avg_kl = total_kl_loss / n
avg_mse = total_mse_loss / n
avg_rmse = np.sqrt(avg_mse)
avg_mae = total_mae_loss / n

print("\nTest Set Evaluation Metrics:")
print(f"Average KL Divergence (Distribution): {avg_kl:.4f}")
print(f"Average MSE (Count): {avg_mse:.4f}")
print(f"Average RMSE (Count): {avg_rmse:.4f}")
print(f"Average MAE (Count): {avg_mae:.4f}")

# -------------------------------
# Visualization: Scatter Plot of Actual vs. Predicted Bag Count
# -------------------------------
# Evaluate on individual flights from test set for count predictions:
actual_counts = []
predicted_counts = []
for idx in range(len(test_dataset)):
    input_vec, target_dist, target_count = test_dataset[idx]
    aircraft_type = torch.tensor([0], dtype=torch.long, device=device)
    day_of_week = torch.tensor([int(input_vec[4].item())], dtype=torch.long, device=device)
    continuous_feats = torch.tensor([[input_vec[1].item(), input_vec[0].item(),
                                        input_vec[3].item(), input_vec[2].item()]], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, count_pred = model(aircraft_type, day_of_week, continuous_feats)
    actual_counts.append(target_count.item())
    predicted_counts.append(count_pred.item())

plt.figure(figsize=(8, 6))
plt.scatter(actual_counts, predicted_counts, alpha=0.5, color='b', edgecolors='k')
plt.plot([min(actual_counts), max(actual_counts)], [min(actual_counts), max(actual_counts)], 'r--', label="Ideal")
plt.xlabel("Actual Total Bag Count")
plt.ylabel("Predicted Total Bag Count")
plt.title("Actual vs. Predicted Total Bag Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
scatter_path = os.path.join("predictions_samples", "actual_vs_predicted_count.png")
plt.savefig(scatter_path)
print(f"Scatter plot saved to {scatter_path}")
plt.show()

# -------------------------------
# Visualization: Boxplot of Count Errors
# -------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(y=count_errors, color="skyblue")
plt.title("Distribution of Absolute Count Errors (Test Set)")
plt.ylabel("Absolute Error in Bag Count")
boxplot_path = os.path.join("predictions_samples", "count_error_boxplot.png")
plt.savefig(boxplot_path)
print(f"Boxplot saved to {boxplot_path}")
plt.show()

# -------------------------------
# Comprehensive Summary
# -------------------------------
summary = {
    "Total Test Samples": n,
    "Average KL Divergence": avg_kl,
    "Average MSE": avg_mse,
    "Average RMSE": avg_rmse,
    "Average MAE": avg_mae,
    "Count Error Std Dev": np.std(count_errors),
    "Count Error Median": np.median(count_errors),
    "Count Error Max": np.max(count_errors)
}
summary_df = pd.DataFrame([summary])
summary_csv = "test_set_evaluation_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print("\nDetailed Evaluation Summary:")
print(summary_df)
print(f"Summary saved to {summary_csv}")
