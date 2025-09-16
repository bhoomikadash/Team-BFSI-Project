# %%
import pandas as pd

#Load the dataset and printing the first few rows 
df = pd.read_csv("fraud_detection_dataset_LLM.csv")

print(df.head())
print(df.info())

# %%
import pandas as pd
import numpy as np

try:
    df = pd.read_csv("fraud_detection_dataset_LLM.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    # Exit the script or handle the error as needed
    exit()

#Handling missing values

print("\n--- Missing Values Count per Column ---")
missing_values_count = df.isnull().sum()
print(missing_values_count)

print("\n--- Missing Values Percentage per Column ---")
missing_values_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_values_percentage)

print("\n--- Any Missing Values in DataFrame? ---")
print(df.isnull().values.any())

print("\n--- Total Missing Values in DataFrame ---")
total_missing_values = df.isnull().sum().sum()
print(total_missing_values)

print("Missing values before handling:")
print(df.isnull().sum())

df.fillna({"kyc_verified": "No"}, inplace=True)

df.dropna(subset=['transaction_amount'], inplace=True)

# Verify that missing values have been handled
print("\nMissing values after handling:")
print(df.isnull().sum())

# %%
# Check for the total number of duplicate transactions
print("Number of duplicate transactions:", df.duplicated(subset=['transaction_id']).sum())

# %%
# Remove duplicate rows based on the 'transaction_id'
df.drop_duplicates(subset=['transaction_id'], inplace=True)
print("Number of duplicate transactions after removal:", df.duplicated(subset=['transaction_id']).sum())

# %%
# Standardize columns
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

num_invalid_timestamps = df['timestamp'].isna().sum()
if num_invalid_timestamps > 0:
    print(f"Warning: {num_invalid_timestamps} invalid timestamps detected and converted to NaT.")

# Normalize categorical values in the 'channel' column 
df['channel'] = df['channel'].str.title()

# Ensure 'transaction_amount' column is numeric; coerce errors to NaN
df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')

# Optional: Report any non-numeric values coerced to NaN in 'transaction_amount'
num_invalid_amounts = df['transaction_amount'].isna().sum()
if num_invalid_amounts > 0:
    print(f"Warning: {num_invalid_amounts} invalid transaction amounts detected and converted to NaN.")

# Display updated DataFrame info and sample
print(df.info())
print(df.head())


# %%
import pandas as pd
import numpy as np

#Feature Engineering 
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday
df['month'] = df['timestamp'].dt.month

df['is_high_value'] = (df['transaction_amount'] > 50000).astype(int)

df = df.sort_values(by=['customer_id', 'timestamp'])
df['time_since_last_txn'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds().fillna(0)

df['rolling_avg_txn_amount'] = df.groupby('customer_id')['transaction_amount'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

df['txn_deviation_from_avg'] = np.abs(df['transaction_amount'] - df['rolling_avg_txn_amount'])

df['customer_transaction_count'] = df.groupby('customer_id')['transaction_id'].transform('count')

df['unique_channels_used'] = df.groupby('customer_id')['channel'].transform('nunique')

# Display the new and existing features to verify
print(df.head())
print(df.info())

# %%
from sklearn.preprocessing import StandardScaler

# Scaling numerical features
num_features = [
    'account_age_days', 'transaction_amount', 'hour', 'weekday', 'month',
    'time_since_last_txn', 'rolling_avg_txn_amount', 'txn_deviation_from_avg',
    'customer_transaction_count', 'unique_channels_used'
]

scaler = StandardScaler()

df[num_features] = scaler.fit_transform(df[num_features])

print(df[num_features].agg(['mean', 'std']))


# %%
from sklearn.preprocessing import OneHotEncoder

#Encoding
cat_features = ['kyc_verified', 'channel']

encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

encoded_arrays = encoder.fit_transform(df[cat_features])

encoded_df = pd.DataFrame(
    encoded_arrays,
    columns=encoder.get_feature_names_out(cat_features),
    index=df.index
)

df = df.drop(columns=cat_features)
df = pd.concat([df, encoded_df], axis=1)

# Verify by displaying first few rows
print(df.head())


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Exploratory Data Analysis
sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 24))
plt.tight_layout(pad=5.0)

def reconstruct_channel(row):
    if row['channel_Mobile'] == 1:
        return 'Mobile'
    elif row['channel_Pos'] == 1:
        return 'Pos'
    elif row['channel_Web'] == 1:
        return 'Web'
    else:
        return 'Unknown'

df['channel_reconstructed'] = df.apply(reconstruct_channel, axis=1)


# 1. Distribution of Fraudulent vs. Legitimate Transactions
sns.countplot(x='is_fraud', data=df, ax=axes[0, 0])
axes[0, 0].set_title('1. Distribution of Fraud vs. Legit Transactions', fontsize=14)
axes[0, 0].set_xlabel('Is Fraud (0 = Legit, 1 = Fraud)')
axes[0, 0].set_ylabel('Count')

# 2. Transaction Amount by Class (without outliers)
sns.boxplot(x='is_fraud', y='transaction_amount', data=df, showfliers=False, ax=axes[0, 1])
axes[0, 1].set_title('2. Transaction Amount by Class (No Outliers)', fontsize=14)
axes[0, 1].set_xlabel('Is Fraud')
axes[0, 1].set_ylabel('Transaction Amount')

# 3. Fraud Transactions by Hour of Day
sns.countplot(x='hour', hue='is_fraud', data=df, ax=axes[1, 0])
axes[1, 0].set_title('3. Transaction Fraud by Hour of Day', fontsize=14)
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(title='Is Fraud', labels=['Legitimate', 'Fraud'])

# 4. Fraud Transactions by Channel (using reconstructed channel)
sns.countplot(x='channel_reconstructed', hue='is_fraud', data=df, ax=axes[1, 1])
axes[1, 1].set_title('4. Transaction Fraud by Channel', fontsize=14)
axes[1, 1].set_xlabel('Transaction Channel')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Is Fraud', labels=['Legitimate', 'Fraud'])

# 5. Fraud Rate by Hour of Day
hourly_fraud_rate = df.groupby('hour')['is_fraud'].mean().reset_index()
sns.barplot(x='hour', y='is_fraud', data=hourly_fraud_rate, ax=axes[2, 0])
axes[2, 0].set_title('5. Fraud Rate by Hour of Day', fontsize=14)
axes[2, 0].set_xlabel('Hour of Day')
axes[2, 0].set_ylabel('Fraud Rate')

# 6. Deviation from Average Transaction Amount (Distribution)
sns.violinplot(x='is_fraud', y='txn_deviation_from_avg', data=df, ax=axes[2, 1])
axes[2, 1].set_title('6. Deviation from Avg Transaction by Class', fontsize=14)
axes[2, 1].set_xlabel('Is Fraud')
axes[2, 1].set_ylabel('Deviation from Avg ($)')

# 7. Time Since Last Transaction Distribution by Class
sns.kdeplot(data=df, x='time_since_last_txn', hue='is_fraud', fill=True, common_norm=False, ax=axes[3, 0])
axes[3, 0].set_title('7. Time Since Last Transaction by Class (Scaled)', fontsize=14)
axes[3, 0].set_xlabel('Scaled Time Since Last Transaction')
axes[3, 0].set_ylabel('Density')


# 8. High-Value Fraud Rate by Channel (using reconstructed channel)
high_value_fraud_pivot = pd.pivot_table(df[df['is_high_value'] == 1], values='is_fraud',
                                        index='channel_reconstructed', aggfunc='mean').reset_index()
sns.barplot(x='channel_reconstructed', y='is_fraud', data=high_value_fraud_pivot, ax=axes[3, 1])
axes[3, 1].set_title('8. High-Value Fraud Rate by Channel', fontsize=14)
axes[3, 1].set_xlabel('Channel')
axes[3, 1].set_ylabel('Fraud Rate for High-Value Txns')

plt.show()


# %%
import os

# Create the 'data/processed' directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save the processed DataFrame to a new CSV file
# The index=False argument prevents pandas from writing the DataFrame index as a column
df.to_csv("data/processed/transactions_processed.csv", index=False)

print("Processed data saved successfully to data/processed/transactions_processed.csv")

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# Define features (X) and target (y)
X = df.drop(columns=['transaction_id', 'customer_id', 'timestamp', 'is_fraud'])
y = df['is_fraud']

# Split into training set and test set (80/20 split) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for the final test set
    stratify=y,
    random_state=42
)

print("Initial split:")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print("-" * 30)

# Split the training set into a new training set and validation set (80/20 split) ---
# You can comment out this entire section if you don't need a validation set.
# This will result in your original X_train and y_train being your final training data.

# X_train_final, X_val, y_train_final, y_val = train_test_split(
#     X_train, y_train,
#     test_size=0.25,      # 25% of the initial 80% is 20% of the total dataset
#     stratify=y_train,
#     random_state=42
# )

# After this split, your data distribution will be approximately 60% Train, 20% Validation, 20% Test.

# print("Final split (with validation set):")
# print(f"Final training set size: {len(X_train_final)}")
# print(f"Validation set size: {len(X_val)}")
# print(f"Testing set size: {len(X_test)}")
# print("-" * 30)

os.makedirs('data/processed', exist_ok=True)

test_df = pd.concat([X_test, y_test], axis=1)

train_df_final = pd.concat([X_train, y_train], axis=1)

train_df_final.to_csv("data/processed/train.csv", index=False)
# val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Datasets saved to data/processed/ directory.")

# %%



