import pandas as pd
import snowflake.connector
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from fitter import Fitter
from scipy.stats import gamma

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
x = np.linspace(min(data) - 10, max(data) + 10, 1000)

# Fit the distribution to the "data" you extracted earlier
f = Fitter(data, distributions=[
    'norm', 'expon', 'gamma', 'lognorm', 'weibull_min', 'weibull_max', 
    'beta', 'pareto', 'rayleigh', 'triang', 'uniform', 'chi2'
])
f.fit()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, beta, gamma, pareto, expon  # Import the best-fitting distributions

# Extract data from Fitter
best_fit = f.get_best()
print(f"Best Fit Output: {best_fit}")
best_fit_name = list(best_fit.keys())[0]  # Get the best distribution name
 # Get its parameters
best_fit_params = best_fit.get(best_fit_name, [])
print(f"Extracted Weibull Parameters: {best_fit_params}")

print(f"Best Fit: {best_fit_name}")
print(f"Best Fit Parameters: {best_fit_params}")
print(f"Parameter Types: {[type(p) for p in best_fit_params]}")

# Generate x-values for smooth plotting
x = np.linspace(min(data), max(data), 1000)

# Function to compute PDF based on the best fit

# Function to compute PDF based on the best fit
# Function to compute PDF based on the best fit
# Function to compute PDF based on the best fit
def clean_params(params):
    try:
        # Convert all values in params to floats
        params = {key: float(value) for key, value in params.items()}
        return tuple(params.values())  # Convert dictionary values to tuple
    except ValueError as e:
        print(f"Error cleaning parameters: {params}. Exception: {e}")
        return None

# Usage in get_pdf


# Function to compute PDF based on the best fit
def get_pdf(dist_name, params, x_vals):
    params = clean_params(params)
    if params is None:
        print(f"Invalid parameters for distribution {dist_name}, skipping fit plot.")
        return None  # Skip if parameters are invalid
    
    # Check if x_vals and params match the expected lengths
    print(f"x_vals length: {len(x_vals)}")
    
    if dist_name == "weibull_min":
        c, loc, scale = params
        return weibull_min.pdf(x_vals, c, loc, scale)
    elif dist_name == "beta":
        a, b, loc, scale = params
        return beta.pdf(x_vals, a, b, loc, scale)
    elif dist_name == "gamma":
        a, loc, scale = params
        return gamma.pdf(x_vals, a, loc, scale)
    elif dist_name == "pareto":
        b, loc, scale = params
        return pareto.pdf(x_vals, b, loc, scale)
    elif dist_name == "expon":
        loc, scale = params
        return expon.pdf(x_vals, loc, scale)
    else:
        return None  # Handle unknown cases







# Generate x-values for smooth plotting, extending the range slightly beyond the min and max of the data
x = np.linspace(min(data) - 10, max(data) + 10, 1000)

# Compute PDF for the best fit
pdf = get_pdf(best_fit_name, best_fit_params, x)

# Print the shape of pdf to ensure it matches x
print(f"x shape: {x.shape}")
print(f"pdf shape: {np.shape(pdf)}")

# Plot the histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Data Histogram")

# Plot the best-fit distribution
if pdf is not None:
    plt.plot(x, pdf, label=f"{best_fit_name.capitalize()} Fit", color='red', linewidth=2)

# Add labels and title
plt.xlabel("Baggage Processing Time (or relevant metric)")
plt.ylabel("Density")
plt.title(f"Best Fit: {best_fit_name.capitalize()} Distribution")
plt.legend()

# Show the plot
plt.show()
