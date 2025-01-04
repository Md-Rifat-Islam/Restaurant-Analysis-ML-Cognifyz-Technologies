import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

class CuisineClassifierApp:
    def __init__(self, root):
        self.root = root
        self.df = None
        self.clf = None
        self.cuisine_encoder = None

        self.root.title("Cuisine Classification")
        self.root.geometry("400x400")

        self.status_label = tk.Label(root, text="Welcome! Load a dataset to begin.")
        self.status_label.pack(pady=10)

        # Buttons
        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(pady=10)

        self.train_button = tk.Button(root, text="Preprocess & Train Model", command=self.preprocess_and_train)
        self.train_button.pack(pady=10)

        self.evaluate_button = tk.Button(root, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict Cuisine", command=self.predict_cuisine)
        self.predict_button.pack(pady=10)

    def load_dataset(self):
        """ Load dataset using file dialog """
        file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.status_label.config(text=f"Dataset loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {e}")

    def preprocess_and_train(self):
        """ Preprocess data and train model """
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return

        # Preprocess the data
        self.df.dropna(subset=['Cuisines'], inplace=True)
        self.df.fillna({'Average Cost for two': self.df['Average Cost for two'].median(),
                        'Aggregate rating': self.df['Aggregate rating'].mean()}, inplace=True)

        binary_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
        for col in binary_columns:
            self.df[col] = self.df[col].apply(lambda x: 1 if x == 'Yes' else 0)

        self.df = pd.get_dummies(self.df, columns=['Currency', 'Rating color', 'Rating text', 'Country Code', 'City'], drop_first=True)

        # Encode target variable
        self.cuisine_encoder = LabelEncoder()
        self.df['Cuisines'] = self.cuisine_encoder.fit_transform(self.df['Cuisines'])

        X = self.df.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Cuisines'], axis=1)
        y = self.df['Cuisines']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clf.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.status_label.config(text=f"Model trained successfully with accuracy: {acc:.2f}")
        joblib.dump(self.clf, 'cuisine_classifier.pkl')

    def evaluate_model(self):
        """ Evaluate the trained model """
        if self.clf is None:
            messagebox.showerror("Error", "Please train the model first!")
            return

        # Prepare the data for evaluation
        X_test, y_test = self.preprocess_test_data()
        y_pred = self.clf.predict(X_test)

        # Classification report and accuracy
        report = classification_report(y_test, y_pred, target_names=self.cuisine_encoder.classes_)
        acc = accuracy_score(y_test, y_pred)

        # Display results
        result = f"Accuracy: {acc:.2f}\n\nClassification Report:\n{report}"
        messagebox.showinfo("Model Evaluation", result)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.cuisine_encoder.classes_, yticklabels=self.cuisine_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def preprocess_test_data(self):
        """ Prepare test data for evaluation """
        X = self.df.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Cuisines'], axis=1)
        y = self.df['Cuisines']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_test, y_test

    def predict_cuisine(self):
        """ Predict cuisine based on the trained model """
        if self.clf is None:
            messagebox.showerror("Error", "Please train the model first!")
            return

        # Example input (replace with real user input or GUI input data)
        example_input = self.df.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Cuisines'], axis=1).iloc[0].values.reshape(1, -1)
        predicted = self.clf.predict(example_input)
        predicted_cuisine = self.cuisine_encoder.inverse_transform(predicted)[0]
        self.status_label.config(text=f"Predicted Cuisine: {predicted_cuisine}")

# Create the main application window
root = tk.Tk()
app = CuisineClassifierApp(root)
root.mainloop()
