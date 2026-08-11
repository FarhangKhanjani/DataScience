import pandas as pd
import numpy as np

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Extract features and target variable
X_train = train_data[['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']]
y_train = train_data['SalePrice']

X_test = test_data[['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']]
y_test = test_data['SalePrice']

# Normalize the data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Add a column of ones to X_train and X_test for the intercept term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Fit the linear regression model using the normal equation
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict the prices
train_predictions = X_train @ theta
test_predictions = X_test @ theta

# Calculate the train and test error (mean squared error)
train_error = np.mean((train_predictions - y_train) ** 2)
test_error = np.mean((test_predictions - y_test) ** 2)

# Output the results
print(f"Regression coefficients: {theta}")
print(f"Train error: {train_error}")
print(f"Test error: {test_error}")

# Save the regression coefficients to a CSV file
np.savetxt('regression_coefficients.csv', theta, delimiter=',')

print("Task 2 completed: Model fitted, coefficients and errors calculated.")
