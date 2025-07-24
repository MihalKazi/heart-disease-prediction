import pandas as pd

# Load your CSV file
df = pd.read_csv('heart.csv')

# Remove the 'condition' column (if it exists)
if 'condition' in df.columns:
    df = df.drop('condition', axis=1)
    print("Removed 'condition' column")
else:
    print("'condition' column not found")

# Save the cleaned file
df.to_csv('heart_cleaned.csv', index=False)
print("Saved cleaned file as 'heart_cleaned.csv'")

# Display the columns in the cleaned file
print(f"Remaining columns: {list(df.columns)}")
print(f"Number of rows: {len(df)}")