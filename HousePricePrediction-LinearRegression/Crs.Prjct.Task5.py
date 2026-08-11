import pandas as pd
import numpy as np

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Compute the average of the prices
price_threshold = train_data['SalePrice'].mean()

# Create a new column for the price label (1 if expensive, 0 if affordable)
train_data['PriceLabel'] = (train_data['SalePrice'] >= price_threshold).astype(int)
test_data['PriceLabel'] = (test_data['SalePrice'] >= price_threshold).astype(int)

# Extract features and target variable
X_train = train_data[['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']]
y_train = train_data['PriceLabel']

X_test = test_data[['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']]
y_test = test_data['PriceLabel']

# Normalize the data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Add a column of ones for the intercept term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize weights
theta = np.zeros(X_train.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (1/m) * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))
    return cost

# Gradient descent for logistic regression
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []
    
    for i in range(num_iterations):
        gradient = (1/m) * X.T @ (sigmoid(X @ theta) - y)
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

# Make predictions
train_predictions = sigmoid(X_train @ theta) >= 0.5
test_predictions = sigmoid(X_test @ theta) >= 0.5

# Calculate the accuracy
train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

# Output the results
print(f"Threshold price: {price_threshold}")
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Save the logistic regression coefficients to a CSV file
np.savetxt('logistic_regression_coefficients.csv', theta, delimiter=',')

# Save the modified train_data to a new CSV file
train_data.to_csv('train_data_with_labels.csv', index=False)

print("Task 5 completed: Logistic regression model fitted, accuracy calculated, and data saved.")
