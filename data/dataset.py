import pandas as pd

# Load dataset
df1 = pd.read_csv("data/german_credit_data.csv")
df2 = pd.read_csv("data/UCI_Credit_Card.csv")
df3 = pd.read_csv("data/cs-training.csv")
# Display first 5 rows
print("First 5 rows of dataset:")
print(df1.head())
print(df2.head())
print(df3.head())
# Display shape
print("\nDataset shape:")
print(df1.shape)
print(df2.shape)
print(df3.shape)

# Display column names
print("\nColumn names:")
print(df1.columns)
print(df2.columns)
print(df3.columns)

# Display data types
print("\nData types:")
print(df1.dtypes)
print(df2.dtypes)
print(df3.dtypes)

# Check missing values
print("\nMissing values in each column:")
print(df1.isnull().sum())
print(df2.isnull().sum())
print(df3.isnull().sum())