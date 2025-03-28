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
# Expanded Hyperparameter Grid Definition
# -------------------------------
hyperparams_grid = {
    "lr": [0.001, 0.0005, 0.0001],
    "hidden_sizes": [[128, 64], [64, 32]],
    "dropout_rate": [0.2, 0.3],
    "optimizer": ["Adam", "AdamW"],
    "scheduler": [None, "CosineAnnealingLR", "ReduceLROnPlateau"],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [32, 64]
}

grid_keys = list(hyperparams_grid.keys())
combinations = list(itertools.product(*[hyperparams_grid[k] for k in grid_keys]))
print("Total hyperparameter combinations to test:", len(combinations))

# -------------------------------
# Training Settings
# -------------------------------
max_epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "final_training_dataset.parquet"

# -------------------------------
# Load and Split Dataset
# -------------------------------
print("Loading dataset...")
full_dataset = BaggageDataset(dataset_path)
dataset_size = len(full_dataset)
print(f"Total flights: {dataset_size}")
# For grid search, we'll use the same split for all configs.
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")

def get_dataloaders(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def train_one_epoch(model, loader, optimizer, device, criterion_dist, criterion_count, scheduler=None, scheduler_on_plateau=False):
    model.train()
    running_loss = 0.0
    for batch in loader:
        input_vec, target_dist, target_count = batch
        # Construct model inputs:
        # Use dummy aircraft_type: assume 0 for each sample.
        aircraft_type = torch.zeros(input_vec.size(0), dtype=torch.long, device=device)
        day_of_week = input_vec[:, 4].long()  # day_of_week at index 4
        # Continuous features: [distance, seats, hour_of_day, is_international]
        continuous_feats = torch.stack([input_vec[:, 1], input_vec[:, 0],
                                          input_vec[:, 3], input_vec[:, 2]], dim=1).to(device)
        target_dist = target_dist.to(device)
        target_count = target_count.to(device)
        
        optimizer.zero_grad()
        dist_pred, count_pred = model(aircraft_type, day_of_week, continuous_feats)
        loss_dist = criterion_dist(torch.log(dist_pred + 1e-8), target_dist)
        loss_count = criterion_count(count_pred.squeeze(), target_count)
        loss = loss_dist + loss_count
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input_vec.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    if scheduler and not scheduler_on_plateau:
        scheduler.step()
    return epoch_loss

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
# Hyperparameter Grid Search Loop
# -------------------------------
results = []  # to store performance for each configuration

for combo in combinations:
    config = dict(zip(grid_keys, combo))
    print("\nTesting configuration:", config)
    
    # Create DataLoaders with specified batch size.
    train_loader, val_loader, test_loader = get_dataloaders(config["batch_size"])
    
    # Instantiate model: assume input vector has 6 features
    model = FlightBaggageModel(num_aircraft_types=10, num_dayofweek=7,
                               embed_dim_aircraft=4, embed_dim_day=3,
                               hidden_sizes=config["hidden_sizes"],
                               dropout_rate=config["dropout_rate"]).to(device)
    
    # Create optimizer based on config
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # Set up scheduler if specified
    scheduler = None
    scheduler_on_plateau = False
    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        scheduler_on_plateau = True
    
    criterion_dist = nn.KLDivLoss(reduction="batchmean")
    criterion_count = nn.MSELoss()
    
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_dist, criterion_count,
                                     scheduler=scheduler, scheduler_on_plateau=scheduler_on_plateau)
        val_loss = evaluate(model, val_loader, device, criterion_dist, criterion_count)
        if scheduler and scheduler_on_plateau:
            scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
    
    results.append({
        "lr": config["lr"],
        "hidden_sizes": config["hidden_sizes"],
        "dropout_rate": config["dropout_rate"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch
    })
    
    # Save best model state for this configuration
    config_name = f"model_{config['optimizer']}_lr{config['lr']}_hs{'-'.join(map(str, config['hidden_sizes']))}_dr{config['dropout_rate']}_wd{config['weight_decay']}_bs{config['batch_size']}_sch{config['scheduler']}.pt"
    torch.save(best_state, config_name)
    print(f"Saved best model for config {config} as {config_name}")

# Save results to CSV for later analysis
results_df = pd.DataFrame(results)
results_csv = "expanded_hyperparameter_results.csv"
results_df.to_csv(results_csv, index=False)
print("Hyperparameter search complete. Results saved to", results_csv)

# -------------------------------
# Plotting Summary Graphs
# -------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
results_df["hidden_sizes_str"] = results_df["hidden_sizes"].astype(str)
scatter = sns.scatterplot(data=results_df, x="lr", y="best_val_loss", hue="hidden_sizes_str",
                          size="dropout_rate", style="optimizer", palette="deep", s=100)
plt.title("Hyperparameter Search: Learning Rate vs Best Validation Loss")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Best Validation Loss")
plt.xscale("log")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig("expanded_hyperparameter_summary.png")
plt.show()

print("\nSummary of Hyperparameter Search:")
print(results_df)
