import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Extract the parameters and the target variable
parameters = ['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']
y_train = train_data['SalePrice']
y_test = test_data['SalePrice']

# Plot the correlation of each parameter with the price
correlations = train_data[parameters + ['SalePrice']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Identify pairs of parameters with positive correlation with price
positive_pairs = []
for i in range(len(parameters)):
    for j in range(i + 1, len(parameters)):
        if correlations.loc[parameters[i], 'SalePrice'] > 0 and correlations.loc[parameters[j], 'SalePrice'] > 0:
            positive_pairs.append((parameters[i], parameters[j]))

# Function to add new feature and fit the model
def add_feature_and_fit_model(param1, param2):
    # Add new feature (product of two parameters)
    train_data['NewFeature'] = train_data[param1] * train_data[param2]
    test_data['NewFeature'] = test_data[param1] * test_data[param2]
    
    # Extract features and target variable
    X_train = train_data[parameters + ['NewFeature']]
    X_test = test_data[parameters + ['NewFeature']]
    
    # Normalize the data
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    
    # Add a column of ones for the intercept term
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
    
    print(f"Parameters: {param1}, {param2}")
    print(f"Train error: {train_error}")
    print(f"Test error: {test_error}")
    print("Theta:", theta)
    print("-------------------------")

# Analyze the results of this model for at least three different pairs of parameters
for param1, param2 in positive_pairs[:3]:
    add_feature_and_fit_model(param1, param2)

print("Task 4 completed: Correlations plotted and models analyzed with new features.")
