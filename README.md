# ML_Project_Fraudanalysis
ML fraud analysis project using XGBoost

import pandas as pd
import xgboost as xgb
from google.colab import files

columns_to_load = ['MSI', 'D_Zone', 'B_Num', 'Hour', 'R_Flag', 'Status']
historical_data = pd.read_csv('/content/27_trining_data (1).csv', usecols=columns_to_load)
historical_data['MSI'] = historical_data['MSI'].astype(str)

# Create a new column 'DZB_Identifier_hour'
historical_data['DZB_Identifier_hour'] = historical_data['D_Zone'] + '_' + historical_data['B_Num'].astype(str) + '_' + historical_data['Hour'].astype(str)

# Calculate the total Distinct_DZB_Count_hour for each unique combination of 'MSI' and 'DZB_Identifier_hour'
distinct_dzb_count_hour_df = historical_data.groupby(['MSI', 'DZB_Identifier_hour'])['DZB_Identifier_hour'].nunique().reset_index(name='Distinct_DZB_Count_hour')

# Create a new column 'DZB_Identifier' by joining 'D_Zone' and 'B_Num'
historical_data['DZB_Identifier'] = historical_data['D_Zone'] + '_' + historical_data['B_Num'].astype(str)

# Calculate the total Distinct_DZB_Count for each unique combination of 'MSI' and 'DZB_Identifier'
distinct_dzb_count_df = historical_data.groupby(['MSI', 'DZB_Identifier'])['DZB_Identifier'].nunique().reset_index(name='Distinct_DZB_Count')

# Merge Distinct_DZB_Count_hour and Distinct_DZB_Count into historical_data
historical_data = historical_data.merge(distinct_dzb_count_hour_df, on=['MSI'], how='left')
historical_data = historical_data.merge(distinct_dzb_count_df, on=['MSI'], how='left')

# Prepare training data
X = historical_data[['Distinct_DZB_Count_hour', 'Distinct_DZB_Count', 'R_Flag']]  # Use historical_data here

y = historical_data['Status']

# Train XGBoost classifier
clf = xgb.XGBClassifier(random_state=42)
clf.fit(X, y)

# Read new data
new_data = pd.read_excel('/content/pol_ML_Oct4.xlsx',usecols=columns_to_load)
new_data['MSI'] = new_data['MSI'].astype(str)

# Create a new column 'DZB_Identifier_hour'
new_data['DZB_Identifier_hour'] = new_data['D_Zone'] + '_' + new_data['B_Num'].astype(str) + '_' + new_data['Hour'].astype(str)

# Calculate the total Distinct_DZB_Count_hour for each unique combination of 'MSI' and 'DZB_Identifier_hour'
distinct_dzb_count_hour_df = new_data.groupby(['MSI', 'DZB_Identifier_hour'])['DZB_Identifier_hour'].nunique().reset_index(name='Distinct_DZB_Count_hour')

# Create a new column 'DZB_Identifier' by joining 'D_Zone' and 'B_Num'
new_data['DZB_Identifier'] = new_data['D_Zone'] + '_' + new_data['B_Num'].astype(str)

# Calculate the total Distinct_DZB_Count for each unique combination of 'MSI' and 'DZB_Identifier'
distinct_dzb_count_df = new_data.groupby(['MSI', 'DZB_Identifier'])['DZB_Identifier'].nunique().reset_index(name='Distinct_DZB_Count')

# Merge Distinct_DZB_Count_hour and Distinct_DZB_Count into new_data
new_data = new_data.merge(distinct_dzb_count_hour_df, on=['MSI'], how='left')
new_data = new_data.merge(distinct_dzb_count_df, on=['MSI'], how='left')

# Prepare features for new data
X_new = new_data[['Distinct_DZB_Count_hour', 'Distinct_DZB_Count', 'R_Flag']]

# Make predictions on new data
predictions_new_data = clf.predict(X_new)

# Add predictions to new data
new_data['Status'] = predictions_new_data

# Set the thresholds for fraud detection
hourly_threshold = 15
daily_threshold = 30

# Mark fraud cases based on the specified thresholds
new_data['Status'] = new_data.apply(
    lambda row: 2 if row['Distinct_DZB_Count_hour'] > hourly_threshold else (
        1 if row['Distinct_DZB_Count'] > daily_threshold else 0),
    axis=1
)

# Save the resulting DataFrame
new_data.to_csv('/content/pol_ML_OP_Oct4(2).csv', index=False)

# Download the file
files.download('/content/pol_ML_OP_Oct4(2).csv')

