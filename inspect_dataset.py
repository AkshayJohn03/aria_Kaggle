import pandas as pd
import os

# Path to your unzipped dataset folder
data_dir = "hull_data"

# List all files
print("📂 Files in dataset directory:")
for f in os.listdir(data_dir):
    print("-", f)

# Load train file (adjust if names differ)
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")

# Load dataframes
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Show basic info
print("\n📊 Train Shape:", train.shape)
print("📊 Test Shape:", test.shape)

print("\n🔎 Train Columns & Types:")
print(train.dtypes)

print("\n📝 First 5 rows of Train:")
print(train.head())

print("\n📈 Basic Statistics:")
print(train.describe(include="all").T)

# Check missing values
print("\n🚨 Missing Values in Train:")
print(train.isnull().sum())
