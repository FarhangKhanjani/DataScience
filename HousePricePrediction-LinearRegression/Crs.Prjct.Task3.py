import pandas as pd
import numpy as np

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Extract features and target variable
parameters = ['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']
y_train = train_data['SalePrice']
y_test = test_data['SalePrice']

# Function to calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Loop through each parameter
for param in parameters:
    print(f"Analyzing parameter: {param}")
    
    train_errors = []
    test_errors = []
    data_sizes = list(range(20, 101, 20)) + list(range(200, 601, 100))
    
    for size in data_sizes:
        # Select the current size of training data
        X_train = train_data[[param]].iloc[:size]
        X_test = test_data[[param]]
        
        # Normalize the data
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_test.mean()) / X_test.std()
        
        # Add a column of ones for the intercept term
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        
        # Fit the linear regression model using the normal equation
        theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train.iloc[:size]
        
        # Predict the prices
        train_predictions = X_train @ theta
        test_predictions = X_test @ theta
        
        # Calculate the train and test error
        train_error = mean_squared_error(y_train.iloc[:size], train_predictions)
        test_error = mean_squared_error(y_test, test_predictions)
        
        # Append errors to the lists
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"Data size: {size}, Train error: {train_error}, Test error: {test_error}")
    
    # Save the errors to CSV files for analysis
    errors_df = pd.DataFrame({
        'Data Size': data_sizes,
        'Train Error': train_errors,
        'Test Error': test_errors
    })
    
    errors_df.to_csv(f'{param}_errors.csv', index=False)
    
    print(f"Completed analysis for parameter: {param}")

print("Task 3 completed: One-parameter models fitted and errors calculated for each parameter.")
