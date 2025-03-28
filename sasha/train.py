import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import copy
from dataset_loader import BaggageDataset
from model import FlightBaggageModel

# Hyperparameters
batch_size = 32
max_epochs = 50
patience = 5
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "final_training_dataset.parquet"

# Load dataset and split it
dataset = BaggageDataset(dataset_path)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, Test: {len(test_dataset)} samples.")

# Instantiate the model (assume 10 aircraft types, 7 days a week)
model = FlightBaggageModel(num_aircraft_types=10, num_dayofweek=7,
                           embed_dim_aircraft=4, embed_dim_day=3,
                           hidden_sizes=[64, 32], dropout_rate=0.2).to(device)

criterion_dist = nn.KLDivLoss(reduction="batchmean")  # expects log-probabilities
criterion_count = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        input_vec, target_dist, target_count = batch
        # Construct inputs for the model:
        # Dummy aircraft_type: using 0 for each sample.
        aircraft_type = torch.zeros(input_vec.size(0), dtype=torch.long, device=device)
        day_of_week = input_vec[:, 4].long()  # using the 5th element as day_of_week
        continuous_feats = torch.stack([input_vec[:, 1], input_vec[:, 0], input_vec[:, 3], input_vec[:, 2]], dim=1).to(device)
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
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_vec, target_dist, target_count = batch
            aircraft_type = torch.zeros(input_vec.size(0), dtype=torch.long, device=device)
            day_of_week = input_vec[:, 4].long()
            continuous_feats = torch.stack([input_vec[:, 1], input_vec[:, 0], input_vec[:, 3], input_vec[:, 2]], dim=1).to(device)
            target_dist = target_dist.to(device)
            target_count = target_count.to(device)
            
            dist_pred, count_pred = model(aircraft_type, day_of_week, continuous_feats)
            loss_dist = criterion_dist(torch.log(dist_pred + 1e-8), target_dist)
            loss_count = criterion_count(count_pred.squeeze(), target_count)
            loss = loss_dist + loss_count
            running_loss += loss.item() * input_vec.size(0)
    return running_loss / len(loader.dataset)

best_val_loss = float("inf")
epochs_no_improve = 0
best_model_state = None

for epoch in range(1, max_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_model_state)
test_loss = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
