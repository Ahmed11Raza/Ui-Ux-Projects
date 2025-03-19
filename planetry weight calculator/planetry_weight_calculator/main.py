import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_weight(weight_on_earth, gravity_ratio):
    """
    Calculate weight on another celestial body based on Earth weight and gravity ratio.
    
    Parameters:
    weight_on_earth (float): Weight on Earth in kg
    gravity_ratio (float): Ratio of celestial body's gravity to Earth's gravity
    
    Returns:
    float: Weight on the celestial body in kg
    """
    return weight_on_earth * gravity_ratio

def main():
    """
    Main function to run the Streamlit application for planetary weight calculation.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Planetary Weight Calculator",
        page_icon="🪐",
        layout="wide"
    )
    
    # Application title and description
    st.title("Planetary Weight Calculator")
    st.write("""
    This application calculates your weight on different celestial bodies in our solar system.
    Enter your weight on Earth and see how it changes across various planets and moons.
    """)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Input Parameters")
        weight_on_earth = st.number_input(
            "Enter your weight on Earth (kg):",
            min_value=0.1,
            max_value=500.0,
            value=70.0,
            step=0.1,
            help="Valid range: 0.1 kg to 500.0 kg"
        )
        
        # Weight unit selection
        unit = st.selectbox(
            "Select output unit:",
            options=["Kilograms (kg)", "Pounds (lb)"],
            index=0
        )
        
        # About section in sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("About")
        st.sidebar.info(
            """
            This calculator uses the gravitational ratios of different celestial bodies 
            relative to Earth to determine how your weight would change in different 
            environments.
            
            **Formula used:**  
            Weight on Planet = Weight on Earth × Gravity Ratio
            """
        )
    
    # Define celestial bodies data
    celestial_bodies = {
        "Mercury": 0.38,
        "Venus": 0.91,
        "Earth": 1.00,
        "Moon": 0.166,
        "Mars": 0.38,
        "Jupiter": 2.34,
        "Saturn": 1.06,
        "Uranus": 0.92,
        "Neptune": 1.19,
        "Pluto": 0.06
    }
    
    # Create dataframe for results
    results = []
    for body, gravity in celestial_bodies.items():
        weight = calculate_weight(weight_on_earth, gravity)
        
        # Convert to pounds if selected
        if unit == "Pounds (lb)":
            weight = weight * 2.20462
            unit_symbol = "lb"
        else:
            unit_symbol = "kg"
            
        results.append({
            "Celestial Body": body,
            "Gravity Ratio": gravity,
            f"Weight ({unit_symbol})": round(weight, 2)
        })
    
    results_df = pd.DataFrame(results)
    
    # Main content area - divided into two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Weight Comparison Table")
        st.dataframe(results_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.header("Weight Visualization")
        
        # Create bar chart visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        bodies = results_df["Celestial Body"]
        weights = results_df[f"Weight ({unit_symbol})"]
        
        # Set colors based on weight comparison to Earth
        earth_weight = weights[bodies == "Earth"].values[0]
        colors = ['#3498db' if w <= earth_weight else '#e74c3c' for w in weights]
        
        # Create bar chart
        bars = ax.bar(bodies, weights, color=colors)
        
        # Customize plot
        ax.set_xlabel("Celestial Body")
        ax.set_ylabel(f"Weight ({unit_symbol})")
        ax.set_title(f"Your Weight Across the Solar System")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Add explanatory text
        st.info("""
        **Interpretation:**
        - Blue bars indicate weights less than or equal to your Earth weight
        - Red bars indicate weights greater than your Earth weight
        """)
    
    # Additional information section
    st.markdown("---")
    st.header("Interesting Facts")
    
    # Display interesting facts based on user's weight
    max_weight_body = results_df.loc[results_df[f"Weight ({unit_symbol})"].idxmax()]["Celestial Body"]
    min_weight_body = results_df.loc[results_df[f"Weight ({unit_symbol})"].idxmin()]["Celestial Body"]
    
    st.write(f"🔹 You would feel heaviest on **{max_weight_body}** and lightest on **{min_weight_body}**.")
    st.write(f"🔹 On the Moon, you would be able to jump approximately 6 times higher than on Earth.")
    st.write(f"🔹 Despite being the largest planet, Jupiter doesn't have the strongest gravity at its surface because it's less dense than Earth.")

if __name__ == "__main__":
    main()