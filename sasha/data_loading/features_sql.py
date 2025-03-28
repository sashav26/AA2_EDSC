import snowflake.connector
import pandas as pd

query_flight_features = """
SELECT
    CONCAT(OPERAT_FLIGHT_NBR, '_', SCHD_LEG_DEP_GMT_TMS) AS Unique_Label,
    SCHD_LEG_DEP_GMT_TMS,
    SCHD_LEG_DEP_LCL_TMS,
    SCHD_LEG_ARVL_AIRPRT_IATA_CD AS Arrival_Airport,
    MILE_GREAT_CIRCLE_DISTANC_QTY AS Distance,
    FLIGHT_LEG_TTL_AVAIL_SEAT_CT AS Seats,
    INTERNATIONAL AS Is_International,
    SCHD_AIRCAFT_EQUIP_CD AS Aircraft_Type,
    FLEET_CD AS Fleet,
    BODYTYPE,
    EXTRACT(HOUR FROM SCHD_LEG_DEP_LCL_TMS) AS Hour_Of_Day,
    EXTRACT(DAYOFWEEK FROM SCHD_LEG_DEP_LCL_TMS) AS Day_Of_Week,
    EXTRACT(MONTH FROM SCHD_LEG_DEP_LCL_TMS) AS Month
FROM LOCAL_DATABASE.ORAAUE.BAGROOM_ARRIVAL
GROUP BY ALL;
"""

conn = snowflake.connector.connect(
    user='EMORY_spring2025_group02',
    password='EDSC2025spring',
    account='aa-itor-universities',
    warehouse='UNIVERSITY_READER',
    database='LOCAL_DATABASE',
    schema='ORAAUE'
)

print("Querying flight-level features...")
cur = conn.cursor()
cur.execute(query_flight_features)
df_features = cur.fetch_pandas_all()
cur.close()
conn.close()
print(f"Retrieved {len(df_features)} flight-level rows.")

# Save to Parquet
df_features.to_parquet("flight_features.parquet", index=False)
print("Saved as flight_features.parquet")
