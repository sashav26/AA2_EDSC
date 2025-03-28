import snowflake.connector
import pandas as pd

# === Step 1: Connect to Snowflake ===
conn = snowflake.connector.connect(
    user='EMORY_spring2025_group02',
    password='EDSC2025spring',
    account='aa-itor-universities',
    warehouse='UNIVERSITY_READER',
    database='LOCAL_DATABASE',
    schema='ORAAUE'
)

# === Step 2: Define SQL to Get Binned Bag Distributions ===
query = """
WITH BagTimes AS (
    SELECT 
        CONCAT(OPERAT_FLIGHT_NBR, '_', SCHD_LEG_DEP_GMT_TMS) AS Unique_Label,
        OPERAT_FLIGHT_NBR,
        SCHD_LEG_DEP_GMT_TMS,
        DATEDIFF(MINUTE, BAG_SCAN_UTC_TMS, SCHD_LEG_DEP_GMT_TMS) AS mins_before_dep
    FROM LOCAL_DATABASE.ORAAUE.BAGROOM_ARRIVAL
    WHERE DATEDIFF(MINUTE, BAG_SCAN_UTC_TMS, SCHD_LEG_DEP_GMT_TMS) BETWEEN 0 AND 450
),
Binned AS (
    SELECT
        Unique_Label,
        OPERAT_FLIGHT_NBR,
        SCHD_LEG_DEP_GMT_TMS,
        FLOOR(mins_before_dep / 15) * 15 AS Time_Bucket,
        COUNT(*) AS Bags_In_Bucket
    FROM BagTimes
    GROUP BY 1, 2, 3, 4
),
TotalBags AS (
    SELECT
        Unique_Label,
        COUNT(*) AS Total_Bags
    FROM BagTimes
    GROUP BY 1
)
SELECT 
    b.Unique_Label,
    b.OPERAT_FLIGHT_NBR,
    b.SCHD_LEG_DEP_GMT_TMS,
    b.Time_Bucket,
    b.Bags_In_Bucket,
    t.Total_Bags,
    b.Bags_In_Bucket / t.Total_Bags AS Bucket_Proportion
FROM Binned b
JOIN TotalBags t ON b.Unique_Label = t.Unique_Label;
"""

# === Step 3: Execute SQL and Fetch Data into DataFrame ===
print("Querying Snowflake...")
cur = conn.cursor()
cur.execute(query)
df = cur.fetch_pandas_all()
cur.close()
conn.close()
print(f"Retrieved {len(df)} rows.")

# === Step 4: Save Locally ===
# Option 1: CSV (compressed)
#df.to_csv("flight_binned_distribution.csv", index=False, compression="gzip")
#print("Saved as flight_binned_distribution.csv.gz")

# Option 2: Parquet (recommended for performance/size)
df.to_parquet("flight_binned_distribution.parquet", index=False)
print("Saved as flight_binned_distribution.parquet")
