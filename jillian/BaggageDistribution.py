import pandas as pd
import snowflake.connector

# Connect to your database
conn = snowflake.connector.connect(
    account='aa-itor-universities',
    user='EMORY_spring2025_group02',
    password='EDSC2025spring',
    warehouse='UNIVERSITY_READER',
    database='LOCAL_DATABASE',
    schema='LOCAL_DATABASE.INFORMATION_SCHEMA'
)


# Execute SQL query
query = """
WITH Time_Buckets AS (
    SELECT 
        CONCAT(OPERAT_FLIGHT_NBR, '_', SCHD_LEG_DEP_GMT_TMS) AS Unique_Label,
        BAG_SCAN_UTC_TMS,
        SCHD_LEG_DEP_GMT_TMS,
        TIMESTAMPDIFF(MINUTE, BAG_SCAN_UTC_TMS, SCHD_LEG_DEP_GMT_TMS) AS Time_Diff,
        FLOOR(TIMESTAMPDIFF(MINUTE, BAG_SCAN_UTC_TMS, SCHD_LEG_DEP_GMT_TMS) / 15) * 15 AS Time_Bucket
    FROM LOCAL_DATABASE.ORAAUE.BAGROOM_ARRIVAL
    WHERE CONCAT(OPERAT_FLIGHT_NBR, '_', SCHD_LEG_DEP_GMT_TMS) = '2347_2024-05-20 03:53:00.000'
)
SELECT 
    Time_Bucket,
    COUNT(*) AS ENTRY_COUNT
FROM Time_Buckets
GROUP BY Time_Bucket
ORDER BY Time_Bucket;

"""

# Fetch data into a DataFrame
df = pd.read_sql(query, conn)

conn.close()

# Extract the "Entry_Count" column as a list
data = df['ENTRY_COUNT'].tolist()

# Import the fitter library
from fitter import Fitter
import matplotlib.pyplot as plt

# Fit the distribution to the "data" you extracted earlier
f = Fitter(data, distributions=['norm', 'expon', 'lognorm']) # Specify simpler distributions
f.fit()

# Print the best distribution
print(f.get_best())

# Plot the distribution
f.summary()



# Add title and labels manually
plt.xlabel("Distributions")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Best Fit Distribution for Baggage Processing Times")

# Show plot
plt.show()

