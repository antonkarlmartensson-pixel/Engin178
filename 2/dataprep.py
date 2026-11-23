import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

"""
This script reads the original train.csv and test.csv files, combines them, and splits them into new train_new.csv and test_new.csv files.
"""

base_dir = Path(__file__).parent

data_dir = base_dir/"data"
data_dir_new = base_dir/"data_new"

train_df = pd.read_csv(data_dir/"train.csv")
test_df = pd.read_csv(data_dir/"test.csv")

combined_df = pd.concat([train_df, test_df], ignore_index = True)

train_new, test_new = train_test_split(
    combined_df,
    train_size = 0.8,
    test_size = 0.2,
    random_state = 42,
    shuffle = True
)

train_new.to_csv(data_dir_new/"train_new.csv", index = False)
test_new.to_csv(data_dir_new/"test_new.csv", index = False)

print(f"train_new.csv: {len(train_new)} rows")
print(f"test_new.csv: {len(test_new)} rows")

"""
It then cleans the data by removing special characters, quotation marks, and whitespace, replacing invalid values with missing values,
removing duplicates, and calculating missing values with the median or mode.

Finally, it saves the cleaned datasets as train_clean.csv and test_clean.csv.
"""

cleaning_columns = train_new.columns.intersection(test_new.columns)

for source_name, df in [
    ("train_new", train_new),
    ("test_new", test_new)]:
    
    df = df.copy()

    # Remove special characters, quotation marks and whitespace
    for column in df.select_dtypes(include = ["object"]).columns:
        df[column] = df[column].replace(r'^\s*$', pd.NA, regex = True)
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].str.replace('"', "", regex = False)
        df[column] = df[column].str.replace(r"\s+", " ", regex = True)

    # Replace invalid employment values with missing values
    if "no_of_days_employed" in df.columns:
        df.loc[df["no_of_days_employed"] > 365000, "no_of_days_employed"] = pd.NA

    # Replace invalid numeric values with missing values
    numeric_rules = {
        "age": (0, 120),
        "no_of_children": (0, 20),
        "migrant_worker": (0, 1),
        "prev_defaults": (0, None),
        "default_in_last_6months": (0, 1),
    }

    for column, (minimum, maximum) in numeric_rules.items():
        if column in df.columns:
            df.loc[df[column] < minimum, column] = pd.NA
            if maximum is not None:
                df.loc[df[column] > maximum, column] = pd.NA

    # Remove exact duplicate records
    df = df.drop_duplicates()

    # Compute remaining missing numeric values with the median
    for column in df.select_dtypes(include=["number"]).columns:
        df[column] = df[column].fillna(df[column].median())

    # Compute remaining missing text values with the mode (expected name, if available, otherwise "Unknown")
    for column in df.select_dtypes(include=["object", "string"]).columns:
        if df[column].isna().any():
            mode = df[column].mode(dropna = True)
            df[column] = df[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    if source_name == "train_new":
        train_clean = df
    else:
        test_clean = df

# Save cleaned datasets
data_dir_new.mkdir(parents = True, exist_ok = True)
train_clean.to_csv(data_dir_new/"train_clean.csv", index = False)
test_clean.to_csv(data_dir_new/"test_clean.csv", index = False)

print(f"train_clean.csv: {len(train_clean)} rows")
print(f"test_clean.csv: {len(test_clean)} rows")
print(f"Remaining missing values in train: {int(train_clean.isna().sum().sum())}")
print(f"Remaining missing values in test: {int(test_clean.isna().sum().sum())}")