import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Example Model: Multi-head network for distribution and total count
class BaggagePredictor(nn.Module):
    def __init__(self, num_depart_airports, num_arrive_airports, 
                 emb_dim=8, input_dim=10, num_time_bins=30, hidden_size=64):
        """
        num_depart_airports: only 2 dep. airports. 1 hot this
        num_arrive_airports: try learned embedding for some spatial type thing? idk if that is applicable
        emb_dim: dimension for destination airport embeddings. would 186 go here???
        input_dim: number of additional features (sin + cos valued params 
        num_time_bins: number of discretized time bins (e.g., 30 for 15-minute bins over 450 mins).
        hidden_size: number of neurons in hidden layers.
        """
        super(BaggagePredictor, self).__init__()
        # Embedding for arriving airport (if needed)
        self.arrive_emb = nn.Embedding(num_arrive_airports, emb_dim)
        
        # Define input layer dimensions (update as per your concatenated features)
        # Assume departing airport is one-hot encoded (2 features) and other features are in input_dim
        total_input_dim = 2 + emb_dim + input_dim
        
        # Shared base layers
        self.fc1 = nn.Linear(total_input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Distribution head: output probabilities for each time bin
        self.dist_head = nn.Linear(hidden_size, num_time_bins)
        
        # Total bag count head: single scalar output
        self.count_head = nn.Linear(hidden_size, 1)
        
    def forward(self, depart_onehot, arrive_idx, other_features):
        """
        depart_onehot: shape (batch_size, 2)
        arrive_idx: shape (batch_size,) integer indices for arriving airport
        other_features: shape (batch_size, input_dim) for other numerical features (time before, sin, cos, seasonality, etc.)
        """
        # Get arrival airport embedding
        arrive_embedded = self.arrive_emb(arrive_idx)
        
        # Concatenate all features
        x = torch.cat([depart_onehot, arrive_embedded, other_features], dim=1)
        
        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Distribution output (softmax for proportions)
        dist_out = F.softmax(self.dist_head(x), dim=1)
        
        # Total bag count output (apply ReLU to ensure non-negativity)
        count_out = F.relu(self.count_head(x))
        
        return dist_out, count_out

# Example training setup
def train(model, dataloader, optimizer, loss_fn_dist, loss_fn_count, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # Example: get batch data, update names as per your data loader
        depart_onehot = batch['depart_onehot'].to(device)
        arrive_idx = batch['arrive_idx'].to(device)
        other_features = batch['other_features'].to(device)
        target_dist = batch['target_dist'].to(device)  # normalized distribution (target)
        target_count = batch['target_count'].to(device)  # total bag count (target)
        
        optimizer.zero_grad()
        pred_dist, pred_count = model(depart_onehot, arrive_idx, other_features)
        
        # Compute losses (customize as needed)
        loss_dist = loss_fn_dist(pred_dist, target_dist)
        loss_count = loss_fn_count(pred_count.squeeze(), target_count)
        
        # Combine losses with weighting (adjust weights as necessary)
        loss = loss_dist + loss_count
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

# Example main function to set up training
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model (update num_arrive_airports and input_dim as per your dataset)
    model = BaggagePredictor(num_depart_airports=2, num_arrive_airports=50, input_dim=10, num_time_bins=30, hidden_size=64)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Define loss functions
    # For distribution, you might use nn.KLDivLoss() or a custom EMD loss
    loss_fn_dist = nn.KLDivLoss(reduction='batchmean')  # example: KL divergence; ensure your inputs are log-probabilities if using this
    loss_fn_count = nn.MSELoss()
    
    # Placeholder: define your DataLoader here
    train_loader = None  # Replace with your DataLoader for training data
    
    # Training loop (simplified)
    for epoch in range(1, 101):
        train_loss = train(model, train_loader, optimizer, loss_fn_dist, loss_fn_count, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        # Here add validation loss computation and early stopping logic

if __name__ == "__main__":
    main()
