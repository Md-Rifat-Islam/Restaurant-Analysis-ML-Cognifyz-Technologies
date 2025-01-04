import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and trained model
file_path = './ML Tasks/Dataset.csv'
data = pd.read_csv(file_path)

# Fill missing values and preprocess dataset
data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna('Unknown', inplace=True)

# Combine features for content-based filtering
data["combined_features"] = data["Cuisines"] + " " + data["City"] + " " + data["Price range"].astype(str)

# Vectorize combined features
cv = CountVectorizer()
feature_matrix = cv.fit_transform(data["combined_features"])

# GUI application class
class RestaurantRatingsApp:
    def __init__(self, master):
        self.master = master
        master.title("Restaurant Recommendation System")
        master.geometry("1000x600")

        # Title
        self.title_label = tk.Label(master, text="Restaurant Recommendation", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=10)

        # User input
        self.criteria_label = tk.Label(master, text="Enter your preferences ", font=("Helvetica", 14, "bold"))
        self.criteria_label.pack(pady=5)
        self.criteria_label_bold = tk.Label(master, text="Cuisine | City | Average Cost for two | (any / multiple)", font=("Helvetica", 12, "bold"))
        self.criteria_label_bold.pack(pady=5)

        self.criteria_entry = tk.Entry(master, width=50)
        self.criteria_entry.pack(pady=10)

        # Button to get recommendations
        self.recommend_button = tk.Button(master, text="Get Recommendations", command=self.get_recommendations)
        self.recommend_button.pack(pady=10)

        # Recommendations display area
        self.recommendations_label = tk.Label(master, text="Recommendations:", font=("Helvetica", 12, "bold"))
        self.recommendations_label.pack(pady=10)

        self.recommendations_text = tk.Text(master, height=15, width=70, state=tk.DISABLED)
        self.recommendations_text.pack(pady=10)

    def parse_user_input(self, user_input):
        user_criteria = user_input.split()
        cuisine = None
        city = None
        price = None

        for item in user_criteria:
            if item.isdigit():
                price = int(item)
            elif item.capitalize() in data["City"].unique():
                city = item.capitalize()
            else:
                cuisine = item.capitalize()
        return cuisine, city, price
    
    def get_recommendations(self):
        user_input = self.criteria_entry.get()

        # Generate feature vector for user input
        user_vector = cv.transform([user_input])
        user_similarity = cosine_similarity(user_vector, feature_matrix)

        # Get indices of top-5 similar restaurants
        similar_indices = user_similarity.argsort()[0][::-1][:5]
        recommendations = data.iloc[similar_indices][["Restaurant Name", "Cuisines", "City", "Average Cost for two", "Aggregate rating"]]

        # Display recommendations
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, "Restaurant Name | Cuisines | City | Average Cost for two | Aggregate rating\n", "bold")
        self.recommendations_text.tag_configure("bold", font=("Helvetica", 10, "bold"))
        if not recommendations.empty:
            for idx, row in recommendations.iterrows():
                self.recommendations_text.insert(tk.END, f"{row['Restaurant Name']} | {row['Cuisines']} | {row['City']} | Cost: {row['Average Cost for two']} | Rating: {row['Aggregate rating']}\n")
        else:
            self.recommendations_text.insert(tk.END, "No recommendations found for the given criteria.")
        self.recommendations_text.config(state=tk.DISABLED)


    # def get_recommendations(self):
    #     user_input = self.criteria_entry.get()
    #     cuisine, city, price = self.parse_user_input(user_input)

    #     # Filter dataset
    #     filtered_data = data.copy()
    #     if cuisine:
    #         filtered_data = filtered_data[filtered_data["Cuisines"].str.contains(cuisine, case=False, na=False)]
    #     if city:
    #         filtered_data = filtered_data[filtered_data["City"].str.contains(city, case=False, na=False)]
    #     if price:
    #         tolerance = 0.1
    #         filtered_data = filtered_data[
    #             (filtered_data["Average Cost for two"] >= price * (1 - tolerance)) &
    #             (filtered_data["Average Cost for two"] <= price * (1 + tolerance))
    #         ]

    #     filtered_data = filtered_data.sort_values(by="Aggregate rating", ascending=False)
    #     recommendations = filtered_data[["Restaurant Name", "Cuisines", "City", "Average Cost for two", "Aggregate rating"]].head(5)

    #     # Display recommendations
    #     self.recommendations_text.config(state=tk.NORMAL)
    #     self.recommendations_text.delete(1.0, tk.END)
    #     self.recommendations_text.insert(tk.END, "Restaurant Name | Cuisines | City | Average Cost for two | Aggregate rating\n", "bold")
    #     self.recommendations_text.tag_configure("bold", font=("Helvetica", 10, "bold"))
    #     if not recommendations.empty:
    #         for idx, row in recommendations.iterrows():
    #             self.recommendations_text.insert(tk.END, f"{row['Restaurant Name']} | {row['Cuisines']} | {row['City']} | Cost: {row['Average Cost for two']} | Rating: {row['Aggregate rating']}\n--------------------------------------------------\n")
    #     else:
    #         self.recommendations_text.insert(tk.END, "No recommendations found for the given criteria.")
    #     self.recommendations_text.config(state=tk.DISABLED)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = RestaurantRatingsApp(root)
    root.mainloop()
