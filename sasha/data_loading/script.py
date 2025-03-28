import pandas as pd

# Load both Parquet files
df_bags = pd.read_parquet("flight_binned_distribution.parquet")
df_flight = pd.read_parquet("flight_features.parquet")

# Merge on Unique_Label
df_merged = pd.merge(df_bags, df_flight, on="UNIQUE_LABEL", how="inner")

# Sanity check
print(f"Merged shape: {df_merged.shape}")
print(df_merged.head(3))

# Save the final model-ready dataset
df_merged.to_parquet("final_training_dataset.parquet", index=False)
print("Saved as final_training_dataset.parquet ✅")
