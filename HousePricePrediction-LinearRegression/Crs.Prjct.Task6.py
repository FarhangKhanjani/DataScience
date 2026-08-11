import pandas as pd
import numpy as np

# Load the previously calculated regression coefficients
theta = np.loadtxt('regression_coefficients.csv', delimiter=',')

# Function to estimate house price
def estimate_price(features):
    # Default values for missing features based on training data statistics
    train_data = pd.read_csv('train_data.csv')
    parameters = ['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']
    means = train_data[parameters].mean()
    stds = train_data[parameters].std()
    
    # Fill missing features with the mean value
    filled_features = {}
    for param in parameters:
        if param in features and features[param] is not None:
            filled_features[param] = features[param]
        else:
            filled_features[param] = means[param]
    
    input_data = np.array([filled_features[param] for param in parameters])
    
    # Normalize the input features
    normalized_input = (input_data - means) / stds
    
    # Add the intercept term
    normalized_input = np.insert(normalized_input, 0, 1)
    
    # Estimate the price
    estimated_price = normalized_input @ theta
    
    return estimated_price

# Function to estimate prices from a file
def estimate_prices_from_file(file_path):
    try:
        input_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found. Please ensure the file path is correct.")
        return None
    except OSError as e:
        print(f"Error opening file: {e}")
        return None
    
    estimated_prices = []
    
    for _, row in input_data.iterrows():
        features = {
            'LotFrontage': row.get('LotFrontage', None),
            'LotArea': row.get('LotArea', None),
            'YearBuilt': row.get('YearBuilt', None),
            'BedroomAbvGr': row.get('BedroomAbvGr', None),
            'KitchenAbvGr': row.get('KitchenAbvGr', None),
            'YrSold': row.get('YrSold', None)
        }
        estimated_price = estimate_price(features)
        estimated_prices.append(estimated_price)
    
    input_data['EstimatedPrice'] = estimated_prices
    return input_data

# Function to manually enter data for multiple houses
def manual_data_entry():
    estimated_prices = []
    while True:
        features = {}
        parameters = ['LotFrontage', 'LotArea', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'YrSold']
        
        for param in parameters:
            value = input(f"Enter {param} (or leave blank if unknown): ")
            if value:
                features[param] = float(value)
            else:
                features[param] = None
        
        estimated_price = estimate_price(features)
        estimated_prices.append(estimated_price)
        print(f"Estimated price: ${estimated_price:.2f}")
        
        cont = input("Do you want to enter another house? (yes/no): ").strip().lower()
        if cont != 'yes':
            break

    return estimated_prices

# Main function to handle user input
def main():
    while True:
        print("Select the input method:")
        print("1. Input data from a file")
        print("2. Input data manually")
        choice = input("Enter 1 or 2: ")
        
        if choice == '1':
            file_path = input("Enter the file path (without quotes): ").strip()
            estimated_data = estimate_prices_from_file(file_path)
            if estimated_data is not None:
                print(estimated_data)
            break
        elif choice == '2':
            estimated_prices = manual_data_entry()
            print("All estimated prices: ", estimated_prices)
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
    print("Task 6 completed: Interactive program for estimating house prices with flexible input handling.")
