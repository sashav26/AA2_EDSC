import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import copy
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import dataset and model classes from your modules
from dataset_loader import BaggageDataset
from model import FlightBaggageModel

# -------------------------------
# Hyperparameter Grid Definition
# -------------------------------
hyperparams_grid = {
    "lr": [0.001, 0.0005, 0.0001],
    "hidden_sizes": [[128, 64], [64, 32]],
    "dropout_rate": [0.2, 0.3]
}
grid_keys = list(hyperparams_grid.keys())
combinations = list(itertools.product(*[hyperparams_grid[k] for k in grid_keys]))

# Training settings
batch_size = 32
max_epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "final_training_dataset.parquet"

# -------------------------------
# Load and Split Dataset
# -------------------------------
print("Loading dataset...")
dataset = BaggageDataset(dataset_path)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def train_one_epoch(model, loader, optimizer, device, criterion_dist, criterion_count):
    model.train()
    running_loss = 0.0
    for batch in loader:
        input_vec, target_dist, target_count = batch
        # Construct model inputs:
        # - Use dummy aircraft_type: assume 0 for each sample (adjust if available).
        aircraft_type = torch.zeros(input_vec.size(0), dtype=torch.long, device=device)
        day_of_week = input_vec[:, 4].long()  # assuming day_of_week is at index 4
        # Continuous features: [distance, seats, hour_of_day, is_international]
        continuous_feats = torch.stack([input_vec[:, 1], input_vec[:, 0],
                                          input_vec[:, 3], input_vec[:, 2]], dim=1).to(device)
        target_dist = target_dist.to(device)
        target_count = target_count.to(device)
        
        optimizer.zero_grad()
        dist_pred, count_pred = model(aircraft_type, day_of_week, continuous_feats)
        # KLDivLoss expects log-probabilities; add epsilon for numerical stability.
        loss_dist = criterion_dist(torch.log(dist_pred + 1e-8), target_dist)
        loss_count = criterion_count(count_pred.squeeze(), target_count)
        loss = loss_dist + loss_count
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input_vec.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device, criterion_dist, criterion_count):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_vec, target_dist, target_count = batch
            aircraft_type = torch.zeros(input_vec.size(0), dtype=torch.long, device=device)
            day_of_week = input_vec[:, 4].long()
            continuous_feats = torch.stack([input_vec[:, 1], input_vec[:, 0],
                                             input_vec[:, 3], input_vec[:, 2]], dim=1).to(device)
            target_dist = target_dist.to(device)
            target_count = target_count.to(device)
            
            dist_pred, count_pred = model(aircraft_type, day_of_week, continuous_feats)
            loss_dist = criterion_dist(torch.log(dist_pred + 1e-8), target_dist)
            loss_count = criterion_count(count_pred.squeeze(), target_count)
            loss = loss_dist + loss_count
            running_loss += loss.item() * input_vec.size(0)
    return running_loss / len(loader.dataset)

# -------------------------------
# Hyperparameter Grid Search
# -------------------------------
results = []  # list to hold performance for each config
print("Starting hyperparameter search over", len(combinations), "combinations...")

for combo in combinations:
    config = dict(zip(grid_keys, combo))
    print("\nTesting configuration:", config)
    
    # Instantiate model: assume input vector has 6 features (for our dataset)
    # Using FlightBaggageModel which expects:
    #   - num_aircraft_types (dummy value, e.g. 10)
    #   - num_dayofweek (7)
    #   - hidden_sizes, dropout_rate from config.
    model = FlightBaggageModel(num_aircraft_types=10, num_dayofweek=7,
                               embed_dim_aircraft=4, embed_dim_day=3,
                               hidden_sizes=config["hidden_sizes"],
                               dropout_rate=config["dropout_rate"]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    criterion_dist = nn.KLDivLoss(reduction="batchmean")
    criterion_count = nn.MSELoss()
    
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_dist, criterion_count)
        val_loss = evaluate(model, val_loader, device, criterion_dist, criterion_count)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered after", epoch, "epochs.")
            break
    
    results.append({
        "lr": config["lr"],
        "hidden_sizes": config["hidden_sizes"],
        "dropout_rate": config["dropout_rate"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch
    })
    
    # Optionally save the best model for this configuration
    config_name = f"model_lr{config['lr']}_hs{'-'.join(map(str,config['hidden_sizes']))}_dr{config['dropout_rate']}.pt"
    torch.save(best_state, config_name)
    print(f"Saved best model for config {config} as {config_name}")

# Save results to CSV for later analysis
results_df = pd.DataFrame(results)
results_csv = "hyperparameter_results.csv"
results_df.to_csv(results_csv, index=False)
print("Hyperparameter search complete. Results saved to", results_csv)

# -------------------------------
# Plotting Summary Graphs
# -------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
# Scatter plot of learning rate vs best_val_loss, using dropout_rate as marker size and hidden_sizes as color (converted to string)
results_df["hidden_sizes_str"] = results_df["hidden_sizes"].astype(str)
scatter = sns.scatterplot(data=results_df, x="lr", y="best_val_loss", hue="hidden_sizes_str", size="dropout_rate", sizes=(50, 200))
plt.title("Hyperparameter Search: Learning Rate vs Best Validation Loss")
plt.xlabel("Learning Rate")
plt.ylabel("Best Validation Loss")
plt.xscale("log")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig("hyperparameter_summary.png")
plt.show()

print("\nSummary of Hyperparameter Search:")
print(results_df)
