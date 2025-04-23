import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import datetime
import base64
from io import BytesIO


class DataLoader:
    """Class responsible for loading and providing the dataset"""
    
    @staticmethod
    @st.cache_data
    def load_data():
        # In a real application, you might load data from a database or API
        # For this example, we'll create sample data
        
        # Sample data: crime incidents in various cities
        data = {
            'latitude': [40.7128, 34.0522, 41.8781, 37.7749, 39.9526, 
                        40.7128, 34.0522, 41.8781, 37.7749, 39.9526],
            'longitude': [-74.0060, -118.2437, -87.6298, -122.4194, -75.1652,
                        -74.0060, -118.2437, -87.6298, -122.4194, -75.1652],
            'location': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Philadelphia',
                        'New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Philadelphia'],
            'category': ['Theft', 'Assault', 'Theft', 'Vandalism', 'Theft',
                       'Traffic Accident', 'Vandalism', 'Traffic Accident', 'Assault', 'Traffic Accident'],
            'datetime': [
                datetime.datetime(2023, 1, 15, 14, 30),
                datetime.datetime(2023, 2, 20, 23, 15),
                datetime.datetime(2023, 3, 5, 10, 45),
                datetime.datetime(2023, 1, 10, 18, 20),
                datetime.datetime(2023, 2, 8, 8, 15),
                datetime.datetime(2023, 4, 12, 13, 0),
                datetime.datetime(2023, 3, 25, 2, 30),
                datetime.datetime(2023, 4, 18, 16, 45),
                datetime.datetime(2023, 5, 7, 14, 10),
                datetime.datetime(2023, 5, 22, 19, 30)
            ],
            'description': [
                'Bicycle stolen from front yard',
                'Altercation outside restaurant',
                'Shoplifting at convenience store',
                'Graffiti on public building',
                'Package theft from porch',
                'Minor collision at intersection',
                'Window broken at business',
                'Vehicle collision with property damage',
                'Dispute escalated to physical altercation',
                'Hit and run incident'
            ],
            'severity': [2, 4, 2, 1, 2, 3, 2, 3, 3, 4]
        }
        
        return pd.DataFrame(data)


class DataFilter:
    """Class responsible for filtering the dataset"""
    
    def __init__(self, df):
        self.df = df
        
    def get_unique_categories(self):
        return list(self.df['category'].unique())
    
    def get_date_range(self):
        date_min = self.df['datetime'].min().date()
        date_max = self.df['datetime'].max().date()
        return date_min, date_max
    
    def filter_data(self, category=None, start_date=None, end_date=None):
        filtered_df = self.df.copy()
        
        # Filter by category
        if category and category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == category]
        
        # Filter by date range
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['datetime'].dt.date >= start_date) & 
                (filtered_df['datetime'].dt.date <= end_date)
            ]
            
        return filtered_df


class DataDownloader:
    """Class responsible for creating download links for data"""
    
    @staticmethod
    def get_csv_download_link(df, filename="filtered_data.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
        return href


class MapVisualizer:
    """Class responsible for creating and displaying the map visualization"""
    
    def __init__(self, df):
        self.df = df
        self.category_colors = {
            'Theft': 'red',
            'Assault': 'darkred',
            'Vandalism': 'orange',
            'Traffic Accident': 'blue'
        }
    
    def create_map(self):
        # Initialize map centered at the mean of the filtered data
        if not self.df.empty:
            map_center = [self.df['latitude'].mean(), self.df['longitude'].mean()]
        else:
            map_center = [0, 0]  # Default center if no data
        
        m = folium.Map(location=map_center, zoom_start=5)
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each data point
        for idx, row in self.df.iterrows():
            # Get color based on category
            color = self.category_colors.get(row['category'], 'gray')
            
            # Create popup content
            popup_content = f"""
            <b>Location:</b> {row['location']}<br>
            <b>Category:</b> {row['category']}<br>
            <b>Date/Time:</b> {row['datetime'].strftime('%Y-%m-%d %H:%M')}<br>
            <b>Description:</b> {row['description']}<br>
            <b>Severity:</b> {row['severity']}
            """
            
            # Add marker
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{row['category']} in {row['location']}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(marker_cluster)
        
        return m


class DataAnalyzer:
    """Class responsible for analyzing and providing statistics about the data"""
    
    def __init__(self, df):
        self.df = df
    
    def get_category_distribution(self):
        cat_counts = self.df['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        return cat_counts
    
    def get_severity_statistics(self):
        if self.df.empty:
            return None
        
        return pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Min', 'Max'],
            'Value': [
                round(self.df['severity'].mean(), 2),
                self.df['severity'].median(),
                self.df['severity'].min(),
                self.df['severity'].max()
            ]
        })


class InteractiveMapApp:
    """Main application class that orchestrates the app components"""
    
    def __init__(self):
        self.setup_page_config()
        self.data_loader = DataLoader()
        self.df = self.data_loader.load_data()
        self.data_filter = DataFilter(self.df)
        self.data_downloader = DataDownloader()
    
    def setup_page_config(self):
        st.set_page_config(page_title="Interactive Map Explorer", layout="wide")
    
    def display_header(self):
        st.title("Interactive Map Explorer")
        st.markdown("""
        This application allows you to explore geographic data on an interactive map.
        You can filter by data category and time range, and download the filtered dataset.
        """)
    
    def display_sidebar_filters(self):
        st.sidebar.header("Data Filters")
        
        # Category filter
        categories = ['All'] + self.data_filter.get_unique_categories()
        selected_category = st.sidebar.selectbox("Select Category", categories)
        
        # Date range filter
        date_min, date_max = self.data_filter.get_date_range()
        selected_date_range = st.sidebar.date_input(
            "Select Date Range",
            [date_min, date_max],
            min_value=date_min,
            max_value=date_max
        )
        
        # Apply filters
        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
        else:
            start_date, end_date = None, None
            
        filtered_df = self.data_filter.filter_data(
            category=selected_category,
            start_date=start_date,
            end_date=end_date
        )
        
        # Display filtered data count
        st.sidebar.write(f"Showing {len(filtered_df)} records")
        
        # Download button for filtered data
        st.sidebar.markdown(
            self.data_downloader.get_csv_download_link(filtered_df),
            unsafe_allow_html=True
        )
        
        return filtered_df
    
    def display_map_and_data(self, filtered_df):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create and display map
            st.subheader("Interactive Map")
            map_visualizer = MapVisualizer(filtered_df)
            m = map_visualizer.create_map()
            folium_static(m, width=800)
        
        with col2:
            # Display data table
            st.subheader("Data Table")
            
            # Format the datetime for display
            display_df = filtered_df.copy()
            display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Show the table with selected columns
            st.dataframe(
                display_df[['location', 'category', 'datetime', 'severity', 'description']],
                height=600
            )
    
    def display_statistics(self, filtered_df):
        st.subheader("Data Statistics")
        
        data_analyzer = DataAnalyzer(filtered_df)
        cat_counts = data_analyzer.get_category_distribution()
        severity_stats = data_analyzer.get_severity_statistics()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Category Distribution:")
            st.table(cat_counts)
        
        with col4:
            if severity_stats is not None:
                st.write("Severity Statistics:")
                st.table(severity_stats)
    
    def display_footer(self):
        st.markdown("---")
        st.markdown("Interactive Map Explorer - Geographic Data Visualization Tool")
    
    def run(self):
        self.display_header()
        filtered_df = self.display_sidebar_filters()
        self.display_map_and_data(filtered_df)
        self.display_statistics(filtered_df)
        self.display_footer()


if __name__ == "__main__":
    app = InteractiveMapApp()
    app.run()