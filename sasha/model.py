import torch
import torch.nn as nn
import torch.nn.functional as F

class FlightBaggageModel(nn.Module):
    def __init__(self, num_aircraft_types, num_dayofweek=7, 
                 embed_dim_aircraft=4, embed_dim_day=3, 
                 hidden_sizes=[64, 32], dropout_rate=0.2):
        super(FlightBaggageModel, self).__init__()
        self.emb_aircraft = nn.Embedding(num_aircraft_types, embed_dim_aircraft)
        self.emb_day = nn.Embedding(num_dayofweek, embed_dim_day)
        # Assume 4 continuous features: [distance, seats, hour_of_day, is_international]
        input_dim = embed_dim_aircraft + embed_dim_day + 4  
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_dim, h))
            self.bn_layers.append(nn.BatchNorm1d(h))
            prev_dim = h
        self.out_distribution = nn.Linear(prev_dim, 31)  # 31 time buckets
        self.out_count = nn.Linear(prev_dim, 1)          
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, aircraft_type, day_of_week, continuous_feats):
        emb_air = self.emb_aircraft(aircraft_type)
        emb_day = self.emb_day(day_of_week)
        x = torch.cat([emb_air, emb_day, continuous_feats], dim=1)
        for linear, bn in zip(self.hidden_layers, self.bn_layers):
            x = linear(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.out_distribution(x)
        dist_probs = F.softmax(logits, dim=1)
        bag_count = self.out_count(x)
        bag_count = F.relu(bag_count)
        return dist_probs, bag_count
