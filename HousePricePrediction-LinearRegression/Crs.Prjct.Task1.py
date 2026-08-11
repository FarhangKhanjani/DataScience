import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Shuffle the data
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate the first 600 entries as test data
test_data = shuffled_data.iloc[:600]
train_data = shuffled_data.iloc[600:]

# Save the test and train data to CSV files
test_data.to_csv('test_data.csv', index=False)
train_data.to_csv('train_data.csv', index=False)

print("Task 1 completed: Data shuffled and separated (The first 600 entries are test data).")
