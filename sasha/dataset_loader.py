import torch
from torch.utils.data import Dataset
import pandas as pd

class BaggageDataset(Dataset):
    """
    PyTorch Dataset for flight-level baggage check-in data.
    
    This dataset is built from a merged Parquet file (e.g. 'final_training_dataset.parquet')
    that contains preprocessed data including:
      - 'unique_label': A unique flight identifier (e.g., OPERAT_FLIGHT_NBR + "_" + SCHD_LEG_DEP_GMT_TMS).
      - 'time_bucket': The 15-minute bucket (e.g., 0, 15, 30, ... up to ~450 minutes before departure).
      - 'bucket_proportion': Proportion of total bags in that time bucket.
      - 'total_bags': The total number of bags for that flight.
      - Flight-level features such as:
            seats, distance, is_international, hour_of_day, day_of_week, month, etc.
    
    Each sample (flight) is constructed by grouping all rows sharing the same 'unique_label'. 
    The flight-level features (assumed identical across a flight's buckets) are extracted from the first row,
    and the ordered bucket proportions are reindexed to a fixed set of time buckets (padding missing buckets with 0).
    """
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.strip().str.lower()
        max_bucket = int(df["time_bucket"].max())
        self.desired_buckets = list(range(0, max_bucket + 15, 15))
        self.groups = df.groupby("unique_label")
        self.flight_data = []
        for unique_label, group in self.groups:
            features = group.iloc[0]
            input_vector = torch.tensor([
                float(features.get("seats", 0)),
                float(features.get("distance", 0)),
                float(features.get("is_international", 0)),
                float(features.get("hour_of_day", 0)),
                float(features.get("day_of_week", 0)),
                float(features.get("month", 0))
            ], dtype=torch.float32)
            bucket_series = pd.Series(
                data=group["bucket_proportion"].astype(float).values, 
                index=group["time_bucket"].astype(int)
            )
            bucket_series = bucket_series.reindex(self.desired_buckets, fill_value=0)
            y_dist = torch.tensor(bucket_series.values, dtype=torch.float32)
            y_total = torch.tensor(float(features.get("total_bags", 0)), dtype=torch.float32)
            self.flight_data.append((input_vector, y_dist, y_total))
    
    def __len__(self):
        return len(self.flight_data)
    
    def __getitem__(self, idx):
        return self.flight_data[idx]
