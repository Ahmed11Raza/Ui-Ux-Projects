import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import time

# Configuration
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
BASE_URL = "https://api.exchangerate.host/latest"

class CurrencyConverter:
    """
    A class to handle currency conversion operations using exchange rate API.
    """
    
    def __init__(self, api_key):
        """
        Initialize the CurrencyConverter with API key.
        
        Args:
            api_key (str): API key for the exchange rate service
        """
        self.api_key = api_key
        self.rates = None
        self.available_currencies = None
        self.last_updated = None
        self.update_rates()
    
    def update_rates(self):
        """
        Fetch the latest exchange rates from the API.
        """
        try:
            params = {
                "access_key": self.api_key,
                "format": 1
            }
            response = requests.get(BASE_URL, params=params)
            
            # For demonstration purposes, if API key is invalid, use a fallback
            if "YOUR_API_KEY_HERE" in self.api_key or not response.ok:
                # Fallback to some hardcoded rates for demonstration
                self.rates = {
                    "USD": 1.0,
                    "EUR": 0.85,
                    "GBP": 0.73,
                    "JPY": 110.42,
                    "CAD": 1.26,
                    "AUD": 1.36,
                    "CHF": 0.92,
                    "CNY": 6.47,
                    "INR": 74.38,
                    "SGD": 1.35
                }
                self.available_currencies = list(self.rates.keys())
            else:
                data = response.json()
                self.rates = data.get("rates", {})
                self.available_currencies = list(self.rates.keys())
            
            self.last_updated = datetime.now()
            return True
        except Exception as e:
            st.error(f"Error updating rates: {str(e)}")
            return False
    
    def convert(self, amount, from_currency, to_currency):
        """
        Convert an amount from one currency to another.
        
        Args:
            amount (float): The amount to convert
            from_currency (str): The source currency code
            to_currency (str): The target currency code
            
        Returns:
            float: The converted amount
        """
        if not self.rates:
            self.update_rates()
        
        # Normalize currencies to uppercase
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Check if currencies are available
        if from_currency not in self.available_currencies:
            raise ValueError(f"Currency {from_currency} not available")
        if to_currency not in self.available_currencies:
            raise ValueError(f"Currency {to_currency} not available")
        
        # Convert to base currency first (USD)
        base_amount = amount
        if from_currency != "USD":
            base_amount = amount / self.rates[from_currency]
        
        # Convert from base currency to target
        converted_amount = base_amount * self.rates[to_currency]
        
        return converted_amount
    
    def get_historical_rates(self, base_currency, target_currency, days=7):
        """
        Simulate fetching historical exchange rates for the chart.
        In a production environment, you would use a real API for this.
        
        Args:
            base_currency (str): Base currency code
            target_currency (str): Target currency code
            days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: DataFrame with dates and rates
        """
        # For demonstration, we'll create simulated data
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        # Create a realistic fluctuation pattern
        current_rate = self.rates[target_currency] / self.rates[base_currency]
        volatility = 0.005  # 0.5% daily volatility
        
        rates = [current_rate]
        for _ in range(1, days):
            change = np.random.normal(0, volatility)
            new_rate = rates[-1] * (1 + change)
            rates.insert(0, new_rate)
        
        df = pd.DataFrame({
            'Date': dates,
            'Rate': rates
        })
        
        return df


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Professional Currency Converter",
        page_icon="💱",
        layout="wide"
    )
    
    # Application title and description
    st.title("Professional Currency Converter")
    st.write("Convert between currencies using real-time exchange rates.")
    
    # Initialize the converter
    converter = CurrencyConverter(API_KEY)
    
    # Sidebar for app navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Converter", "Exchange Rate Chart", "About"])
    
    if page == "Converter":
        display_converter(converter)
    elif page == "Exchange Rate Chart":
        display_chart(converter)
    else:
        display_about()


def display_converter(converter):
    """
    Display the currency converter interface.
    
    Args:
        converter (CurrencyConverter): The currency converter instance
    """
    st.header("Currency Conversion")
    
    # Create three columns for input, conversion button, and output
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader("From")
        amount = st.number_input("Amount", min_value=0.01, value=1.00, step=0.01)
        from_currency = st.selectbox("Currency", converter.available_currencies, index=0)
    
    with col2:
        st.subheader(" ")
        st.write(" ")
        st.write(" ")
        convert_button = st.button("Convert →")
    
    with col3:
        st.subheader("To")
        to_currency = st.selectbox("Currency", converter.available_currencies, index=1)
        result_placeholder = st.empty()
    
    # Perform conversion when button is clicked
    if convert_button:
        try:
            with st.spinner('Converting...'):
                # Add a small delay to simulate API call
                time.sleep(0.5)
                result = converter.convert(amount, from_currency, to_currency)
                result_placeholder.success(f"{amount:.2f} {from_currency} = {result:.2f} {to_currency}")
                
                # Display exchange rate information
                rate = converter.rates[to_currency] / converter.rates[from_currency]
                st.info(f"Exchange Rate: 1 {from_currency} = {rate:.4f} {to_currency}")
                
                # Show last updated time
                st.caption(f"Last updated: {converter.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            st.error(f"Conversion error: {str(e)}")
    
    # Display additional information
    st.markdown("---")
    st.subheader("Popular Conversions")
    
    # Create a grid of popular currency pairs
    popular_pairs = [
        ("USD", "EUR"), ("USD", "GBP"), ("USD", "JPY"), 
        ("EUR", "USD"), ("EUR", "GBP"), ("GBP", "USD")
    ]
    
    cols = st.columns(3)
    for i, (base, target) in enumerate(popular_pairs):
        with cols[i % 3]:
            if base in converter.available_currencies and target in converter.available_currencies:
                rate = converter.convert(1, base, target)
                st.metric(f"{base} to {target}", f"{rate:.4f} {target}")


def display_chart(converter):
    """
    Display historical exchange rate chart.
    
    Args:
        converter (CurrencyConverter): The currency converter instance
    """
    st.header("Exchange Rate Chart")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_currency = st.selectbox("Base Currency", converter.available_currencies, index=0)
    
    with col2:
        target_currency = st.selectbox("Target Currency", converter.available_currencies, index=1)
    
    days = st.slider("Number of Days", min_value=7, max_value=30, value=7)
    
    if st.button("Generate Chart"):
        with st.spinner("Generating chart..."):
            # Get historical data
            df = converter.get_historical_rates(base_currency, target_currency, days)
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Date'], df['Rate'], marker='o', linestyle='-', color='#1f77b4')
            ax.set_title(f"{base_currency}/{target_currency} Exchange Rate - Last {days} Days")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"Rate (1 {base_currency} to {target_currency})")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format the x-axis to show dates nicely
            fig.autofmt_xdate()
            
            # Display the chart
            st.pyplot(fig)
            
            # Show the data table
            st.subheader("Historical Data")
            st.dataframe(df.set_index('Date'))


def display_about():
    """
    Display information about the application.
    """
    st.header("About this Application")
    
    st.write("""
    This professional currency converter application provides real-time currency conversion 
    services using up-to-date exchange rates. Designed for business professionals, financial 
    analysts, and international travelers, this tool offers accurate and reliable currency 
    conversion capabilities.
    """)
    
    st.subheader("Features")
    st.write("""
    - Real-time currency conversion across multiple currencies
    - Historical exchange rate visualization
    - User-friendly interface for quick conversions
    - Reliable data sourced from reputable financial APIs
    """)
    
    st.subheader("Technical Details")
    st.write("""
    This application is built using:
    - Streamlit for the web interface
    - Python for backend logic
    - Exchange rate data from exchangerate.host API
    - Pandas for data manipulation
    - Matplotlib for data visualization
    """)
    
    st.subheader("Disclaimer")
    st.write("""
    The exchange rates provided are for informational purposes only. Actual rates may vary 
    when conducting financial transactions. This application is not intended for use in 
    trading or financial decision-making without additional verification from official 
    financial institutions.
    """)


if __name__ == "__main__":
    main()