# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

import warnings

warnings.filterwarnings('ignore')

# Load the dataset
# file_path = '/home/muhammad/Desktop/Cognifyz Technologies/ML Tasks/Dataset.csv'
file_path = 'ML Tasks/Dataset.csv'
df = pd.read_csv(file_path)

"""# Step 1: Preprocessing

## Step 1.1 : Handle Missing Values
"""

print("Missing Values:\n", df.isnull().sum())
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna('Unknown', inplace=True)

"""## 1.2 Encode categorical variables"""

# Display data types of all columns
print("Data Types of Each Column:")
print(df.dtypes)

# Initialize label encoder
label_encoder = LabelEncoder()

data_frame = df.copy()

# List of columns to encode (example: categorical columns like 'Restaurant Name', 'City')
categorical_columns = ['Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 'Cuisines', 'Currency', 'Has Table booking',
                       'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating color', 'Rating text']

# Apply label encoding to each categorical column
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Data after Label Encoding:")
print(df.head())

"""## 1.3 Feature Selection

### 1.3.1 : Features Correlation Matrix
"""

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
# print("\nCorrelation Matrix:")
# print(correlation_matrix)

target_corr = correlation_matrix['Aggregate rating']

print("\nCorrelation with Target:")
print(target_corr.sort_values())

"""### 1.3.2 : Feature Importance (Tree-Based Models)"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

X0 = df.drop('Aggregate rating', axis=1)
y0 = df['Aggregate rating']

model = RandomForestRegressor()
model.fit(X0, y0)
feature_importance = pd.DataFrame({
    'Feature': X0.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance using Tree-based models:")
print(feature_importance)

# Creating DataFrames
corr_df = pd.DataFrame(target_corr)
importance_df = pd.DataFrame(feature_importance)

print(corr_df.columns)        # To check column names in the correlation DataFrame
print(importance_df.columns)  # To check column names in the importance DataFrame

# Reset the index of correlation DataFrame to create a 'Feature' column
corr_df = corr_df.reset_index()
corr_df.columns = ['Feature', 'Correlation']  # Rename columns to ensure consistency

# Merge the DataFrames
combined_df = corr_df.merge(importance_df, on='Feature')

# Select features with absolute correlation > 0.2 and importance > threshold (e.g., 0.001)
threshold_correlation = 0.3
threshold_importance = 0.002
selected_features = combined_df[
    (combined_df['Correlation'].abs() > threshold_correlation) |
    (combined_df['Importance'] > threshold_importance)
].sort_values(by='Importance', ascending=False)

print(selected_features[['Feature', 'Correlation', 'Importance']], '\n')

# Extract top 3 or 5 features based on importance
top_features_3 = selected_features.head(3)['Feature'].tolist()  # Top 3 features
top_features_5 = selected_features.head(5)['Feature'].tolist()  # Top 5 features

# Choose one based on your requirement
top_features = top_features_5  # or top_features_5 for top 5 features

print("Top Features:", top_features)

# Original DataFrame (X)
selected_features = top_features
X_selected = df[selected_features]

print(X_selected.head())

# Separate features and target variable
X = X_selected.drop('Restaurant ID', axis=1)  # As Restaurant ID is irrevalent
y = df['Aggregate rating']

"""## 1.4 Split the data"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Step 2: Train a Regression Model"""

# Initialize model
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(random_state=42)

# Train models
linear_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

"""# Step 3: Evaluate the Models"""

# Predictions
linear_preds = linear_model.predict(X_test)
dt_preds = decision_tree_model.predict(X_test)
rf_preds = random_forest_model.predict(X_test)

def evaluate_model(name, y_test, preds):
    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Calculate RMSE for better interpretation
    rmse = np.sqrt(mse)

    # Calculate accuracy
    #accuracy = 1 - (sum(abs(y_test - preds)) / sum(y_test))

    # Print metrics
    print(f"{name} --> Mean Squared Error: {mse:.4f} | R-squared: {r2:.4f} | RMSE: {rmse:.4f}")

    # Return metrics for comparison
    return {'name': name, 'MSE': mse, 'R-squared': r2, 'RMSE': rmse}

# Store evaluation results for all models
linear_results = evaluate_model("Linear Regression       ", y_test, linear_preds)
dt_results = evaluate_model("Decision Tree Regression", y_test, dt_preds)
rf_results = evaluate_model("Random Forest Regression", y_test, rf_preds)

# Compare models based on MSE, R-squared, and RMSE
models = [linear_results, dt_results, rf_results]

# Initialize best_model with a very high MSE value
best_model = None
best_score = float('inf')  # Start with the best (lowest) MSE value

# Select the best model based on the metrics
for model in models:
    if model['MSE'] < best_score:  # Select R-squared or MSE or RMSE
        best_score = model['MSE']
        best_model = model

# Output the best model
print(f"The best model is: {best_model['name']}")

"""# Step 4: Interpret Results"""

best_model_name = best_model['name']
if best_model_name == "Linear Regression       ":
    best_model = linear_model
elif best_model_name == "Decision Tree Regression":
    best_model = decision_tree_model
elif best_model_name == "Random Forest Regression":
    best_model = random_forest_model

# Feature Importance for Tree-based models
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(f"\nFeature Importance ({best_model}):")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title(f"\nFeature Importance ({best_model}):")
plt.show()

# Actual vs Predicted Ratings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, alpha=0.6, color='r')
plt.title(f'Actual vs Predicted Ratings ({best_model})')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()

# Extract unique values from the dataset
# Clean values to ensure no special characters or extra spaces
rating_color_values = [str(color).strip() for color in data_frame['Rating color'].unique()]
rating_text_values = [str(text).strip() for text in data_frame['Rating text'].unique()]
price_range_values = sorted([int(price) for price in data_frame['Price range'].unique()])  # Ensure sorted numeric values


import joblib

# Saving the best model to a file
joblib.dump(best_model, 'predict_restaurant_model.pkl')

print("Model saved as 'predict_restaurant_model.pkl'")

import tkinter as tk
from tkinter import ttk, messagebox
import joblib


# Loading the trained model
model = joblib.load("predict_restaurant_model.pkl")

class RestaurantRatingsApp:
    def __init__(self, master):
        self.master = master
        master.title("Restaurant Ratings Prediction App")
        master.geometry("500x600")  # Set the window size

        # Title
        self.label = tk.Label(master, text="Restaurant Ratings Predictor", font=("Helvetica", 16, "bold"))
        self.label.pack(pady=10)

        # Input fields
        self.create_input_field("Votes:", "int")
        self.create_dropdown_field("Rating Color:", rating_color_values)
        self.create_dropdown_field("Rating Text:", rating_text_values)
        self.create_dropdown_field("Price Range:", price_range_values)

        # Predict button
        self.predict_button = tk.Button(master, text="Predict Rating", command=self.predict_rating, font=("Helvetica", 12, "bold"), bg="green", fg="white")
        self.predict_button.pack(pady=20)

        # Result label
        self.result_label = tk.Label(master, text="", font=("Helvetica", 14), fg="blue")
        self.result_label.pack(pady=10)

    def create_input_field(self, label_text, input_type):
        """Create a labeled input field."""
        frame = tk.Frame(self.master)
        frame.pack(pady=5)

        label = tk.Label(frame, text=label_text, font=("Helvetica", 12))
        label.pack(side=tk.LEFT, padx=5)

        entry = tk.Entry(frame, font=("Helvetica", 12))
        entry.pack(side=tk.LEFT, padx=5)

        setattr(self, label_text.lower().replace(" ", "_").replace(":", ""), entry)

    def create_dropdown_field(self, label_text, options):
        """Create a labeled dropdown menu."""
        frame = tk.Frame(self.master)
        frame.pack(pady=5)

        label = tk.Label(frame, text=label_text, font=("Helvetica", 12))
        label.pack(side=tk.LEFT, padx=5)

        # Ensure dropdown options are strings
        options = [str(option) for option in options]

        combo = ttk.Combobox(frame, values=options, font=("Helvetica", 12))
        combo.pack(side=tk.LEFT, padx=5)
        combo.current(0)  # Set default value

        setattr(self, label_text.lower().replace(" ", "_").replace(":", ""), combo)

    def predict_rating(self):
        try:
            # Extract and preprocess user inputs
            votes = int(self.votes.get())  # Convert votes input to integer
            rating_color = self.rating_color.get()
            rating_text = self.rating_text.get()
            price_range = int(self.price_range.get())  # Convert price range to integer

            print(f"Votes: {votes}, Rating Color: {rating_color}, Rating Text: {rating_text}, Price Range: {price_range}")

            # Encode categorical inputs as necessary
            rating_color_encoded = rating_color_values.index(rating_color)
            rating_text_encoded = rating_text_values.index(rating_text)

            # Create input array for the model
            features = np.array([[votes, rating_color_encoded, rating_text_encoded, price_range]])

            # Predict rating
            predicted_rating = model.predict(features)[0]

            # Display result
            self.result_label.config(text=f"Predicted Rating: {predicted_rating:.2f}")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid inputs.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Initialize Tkinter window
root = tk.Tk()
app = RestaurantRatingsApp(root)
root.mainloop()
