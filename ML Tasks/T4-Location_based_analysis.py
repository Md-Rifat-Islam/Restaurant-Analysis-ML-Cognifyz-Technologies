import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# GUI Application Class
class LocationAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Location-Based Analysis")
        self.root.geometry("800x600")

        # UI Elements
        self.label = tk.Label(root, text="Location-Based Analysis", font=("Arial", 16))
        self.label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_dataset, font=("Arial", 12))
        self.load_button.pack(pady=10)

        self.map_button = tk.Button(root, text="Generate Map", command=self.generate_map, font=("Arial", 12), state=tk.DISABLED)
        self.map_button.pack(pady=10)

        self.heatmap_button = tk.Button(root, text="Generate Heatmap", command=self.generate_heatmap, font=("Arial", 12), state=tk.DISABLED)
        self.heatmap_button.pack(pady=10)

        self.stats_button = tk.Button(root, text="Analyze Statistics", command=self.analyze_statistics, font=("Arial", 12), state=tk.DISABLED)
        self.stats_button.pack(pady=10)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit, font=("Arial", 12))
        self.quit_button.pack(pady=10)

        self.status_label = tk.Label(root, text="", font=("Arial", 10), fg="green")
        self.status_label.pack(pady=10)

        # Dataset variable
        self.df = None

    def load_dataset(self):
        # Load dataset using file dialog
        file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.status_label.config(text="Dataset loaded successfully!")
                self.map_button.config(state=tk.NORMAL)
                self.heatmap_button.config(state=tk.NORMAL)
                self.stats_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
        else:
            messagebox.showinfo("Info", "No file selected.")

    def generate_map(self):
        if self.df is not None:
            try:
                # Clean dataset for missing lat/lon
                df = self.df.dropna(subset=['Longitude', 'Latitude'])

                # Create a base map
                restaurant_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)

                # Add markers
                for _, row in df.iterrows():
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=f"{row['Restaurant Name']} ({row['Cuisines']})"
                    ).add_to(restaurant_map)

                # Save the map
                restaurant_map.save("restaurants_map.html")
                messagebox.showinfo("Success", "Map saved as 'restaurants_map.html'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate map: {e}")
        else:
            messagebox.showwarning("Warning", "No dataset loaded.")

    def generate_heatmap(self):
        if self.df is not None:
            try:
                df = self.df.dropna(subset=['Longitude', 'Latitude'])

                # Create a heatmap
                restaurant_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)
                heatmap_data = df[['Latitude', 'Longitude']].dropna()
                HeatMap(heatmap_data.values).add_to(restaurant_map)

                # Save the heatmap
                restaurant_map.save("restaurants_heatmap.html")
                messagebox.showinfo("Success", "Heatmap saved as 'restaurants_heatmap.html'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate heatmap: {e}")
        else:
            messagebox.showwarning("Warning", "No dataset loaded.")

    def analyze_statistics(self):
        if self.df is not None:
            try:
                # Top cities by number of restaurants
                restaurant_count = self.df.groupby('City').size().reset_index(name='Number of Restaurants')
                top_cities = restaurant_count.sort_values('Number of Restaurants', ascending=False).head(10)

                # Plot the top cities
                plt.figure(figsize=(10, 6))
                sns.barplot(data=top_cities, x='Number of Restaurants', y='City', palette='viridis')
                plt.title('Top 10 Cities by Number of Restaurants')
                plt.show()

                # Average rating vs price range
                city_stats = self.df.groupby('City').agg({
                    'Aggregate rating': 'mean',
                    'Price range': 'mean'
                }).reset_index()

                city_stats.rename(columns={'Aggregate rating': 'Average Rating', 'Price range': 'Average Price Range'}, inplace=True)
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=city_stats, x='Average Price Range', y='Average Rating', hue='City', size='Average Rating', sizes=(40, 200))
                plt.title('Average Price Range vs Average Rating by City')
                plt.show()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to analyze statistics: {e}")
        else:
            messagebox.showwarning("Warning", "No dataset loaded.")


# Main Function
if __name__ == "__main__":
    root = tk.Tk()
    app = LocationAnalysisApp(root)
    root.mainloop()
