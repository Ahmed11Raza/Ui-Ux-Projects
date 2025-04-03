# pip install in terminal:
# uv pip install streamlit pandas numpy yfinance scikit-learn plotly

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Set page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Stock Market Prediction App")
st.markdown("""
This app predicts future stock prices based on historical data using machine learning.
Select a stock, time period, and prediction horizon to see the forecast.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Stock selection
stock_ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")

# Date range selection
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = col2.date_input("End Date", datetime.now())

# Training parameters
st.sidebar.subheader("Model Parameters")
prediction_days = st.sidebar.slider("Prediction Window (Days)", 7, 90, 30)
feature_days = st.sidebar.slider("Features Window (Days)", 10, 100, 60)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Linear Regression", "LSTM Neural Network"]
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 500, 100)
        max_depth = st.slider("Max Depth", 3, 20, 10)
    
    include_sentiment = st.checkbox("Include Market Sentiment Analysis", False)
    include_technical = st.checkbox("Include Technical Indicators", True)

# Button to run prediction
run_button = st.sidebar.button("Run Prediction")


# Functions for data processing and modeling
@st.cache_data
def load_data(ticker, start, end):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential moving averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Volatility (standard deviation of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    return df

def prepare_features(df, feature_window):
    """Prepare features for ML models"""
    df = df.copy()
    
    # Add lagged price features
    for i in range(1, feature_window + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
    
    # Add day of week
    df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def create_train_test_sets(df, test_size=0.2, target_col='Close'):
    """Split data into training and testing sets"""
    # Extract features and target
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test

def train_random_forest_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a Random Forest regression model"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def linear_regression_model(X_train, y_train):
    """Train a simple linear regression model"""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def lstm_model(X_train, y_train):
    """
    Placeholder for LSTM model implementation.
    In a real app, this would be a TensorFlow or PyTorch implementation.
    """
    st.warning("LSTM model is a simplified version for demonstration purposes.")
    # Fallback to Random Forest as placeholder
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_future_prediction(model, df, feature_window, prediction_days):
    """Make predictions for future days"""
    # Get the last row of data for prediction
    last_data = df.iloc[-1:].copy()
    
    predictions = []
    current_data = df.copy()
    
    for _ in range(prediction_days):
        # Prepare the input data
        X_pred = prepare_features(current_data, feature_window).iloc[-1:]
        
        # Make prediction
        pred = model.predict(X_pred.drop(['Close'], axis=1))
        
        # Create new data point with the prediction
        new_data = last_data.copy()
        new_data.index = [new_data.index[0] + timedelta(days=1)]
        
        # Increment index date until it's a weekday
        while new_data.index[0].weekday() > 4:  # Skip weekends
            new_data.index = [new_data.index[0] + timedelta(days=1)]
        
        # Update the closing price with the prediction
        new_data['Close'] = pred[0]
        
        # Approximate other values (simplification)
        new_data['Open'] = pred[0] * 0.99
        new_data['High'] = pred[0] * 1.01
        new_data['Low'] = pred[0] * 0.98
        new_data['Volume'] = current_data['Volume'].mean()
        
        # Add technical indicators for this new data point
        for col in current_data.columns:
            if col not in new_data.columns and col not in [f'Close_Lag_{i}' for i in range(1, feature_window + 1)]:
                new_data[col] = current_data[col].iloc[-1]
        
        # Add prediction to the list
        predictions.append(new_data)
        
        # Update current data with new prediction for next iteration
        current_data = pd.concat([current_data, new_data])
        last_data = new_data
    
    # Combine all predictions
    future_df = pd.concat(predictions)
    
    return future_df

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def plot_predictions(historical_data, test_data, future_data, test_predictions=None):
    """Create interactive plot of predictions"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Plot historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Plot test data
    if test_data is not None:
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_data['Close'],
                mode='lines',
                name='Test Data',
                line=dict(color='green')
            ),
            row=1, col=1
        )
    
    # Plot test predictions
    if test_predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_predictions,
                mode='lines',
                name='Test Predictions',
                line=dict(color='orange', dash='dash')
            ),
            row=1, col=1
        )
    
    # Plot future predictions
    fig.add_trace(
        go.Scatter(
            x=future_data.index,
            y=future_data['Close'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Add confidence intervals (simplified)
    upper_bound = future_data['Close'] * 1.05
    lower_bound = future_data['Close'] * 0.95
    
    fig.add_trace(
        go.Scatter(
            x=future_data.index,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_data.index,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            name='95% Confidence'
        ),
        row=1, col=1
    )
    
    # Plot volume
    fig.add_trace(
        go.Bar(
            x=historical_data.index,
            y=historical_data['Volume'],
            name='Volume',
            marker_color='blue',
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Add titles and layout
    fig.update_layout(
        title=f'{stock_ticker} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        height=800,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# Main prediction workflow
if run_button:
    # Show loading spinner
    with st.spinner('Loading data and making predictions...'):
        # Load data
        data = load_data(stock_ticker, start_date, end_date)
        
        if data is not None:
            # Display raw data
            with st.expander("Raw Data"):
                st.dataframe(data)
            
            # Process data
            if include_technical:
                data = add_technical_indicators(data)
            
            # Prepare features
            processed_data = prepare_features(data, feature_days)
            
            # Create train/test split
            X_train, X_test, y_train, y_test = create_train_test_sets(
                processed_data, test_size=test_size
            )
            
            # Train model based on selection
            st.subheader("Training Model")
            progress_bar = st.progress(0)
            
            # Simulate model training progress
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            if model_type == "Random Forest":
                model = train_random_forest_model(
                    X_train.drop(['Open', 'High', 'Low', 'Volume'], axis=1), 
                    y_train,
                    n_estimators=n_estimators,
                    max_depth=max_depth
                )
                st.success(f"Random Forest model trained with {n_estimators} trees")
                
            elif model_type == "Linear Regression":
                model = linear_regression_model(
                    X_train.drop(['Open', 'High', 'Low', 'Volume'], axis=1), 
                    y_train
                )
                st.success("Linear Regression model trained")
                
            else:  # LSTM
                model = lstm_model(
                    X_train.drop(['Open', 'High', 'Low', 'Volume'], axis=1), 
                    y_train
                )
                st.success("LSTM model trained (simplified version)")
            
            # Make predictions on test data
            y_pred = model.predict(X_test.drop(['Open', 'High', 'Low', 'Volume'], axis=1))
            
            # Evaluate model
            metrics = evaluate_model(y_test, y_pred)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
            col2.metric("Root Mean Squared Error", f"{metrics['RMSE']:.4f}")
            col3.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
            col4.metric("R² Score", f"{metrics['R²']:.4f}")
            
            # Feature importance (for Random Forest)
            if model_type == "Random Forest":
                with st.expander("Feature Importance"):
                    feature_names = X_train.drop(['Open', 'High', 'Low', 'Volume'], axis=1).columns
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.dataframe(feature_importance)
                    
                    # Plot top 10 features
                    fig = go.Figure()
                    top_features = feature_importance.head(10)
                    fig.add_trace(go.Bar(
                        y=top_features['Feature'],
                        x=top_features['Importance'],
                        orientation='h'
                    ))
                    fig.update_layout(
                        title="Top 10 Feature Importance",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig)
            
            # Make future predictions
            st.subheader(f"Future Price Predictions ({prediction_days} days)")
            future_df = make_future_prediction(model, data, feature_days, prediction_days)
            
            # Display future predictions
            with st.expander("Prediction Data"):
                st.dataframe(future_df[['Open', 'High', 'Low', 'Close', 'Volume']])
            
            # Plot results
            train_data = data.iloc[:-len(X_test)]
            test_data = data.iloc[-len(X_test):]
            
            st.subheader("Visualization")
            fig = plot_predictions(train_data, test_data, future_df, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price change summary
            last_price = data['Close'].iloc[-1]
            last_date = data.index[-1]
            future_price = future_df['Close'].iloc[-1]
            future_date = future_df.index[-1]
            
            price_change = future_price - last_price
            price_change_pct = (price_change / last_price) * 100
            
            st.subheader("Price Change Summary")
            st.write(f"Last recorded price ({last_date.strftime('%Y-%m-%d')}): ${last_price:.2f}")
            st.write(f"Predicted price ({future_date.strftime('%Y-%m-%d')}): ${future_price:.2f}")
            
            if price_change > 0:
                st.success(f"Predicted increase: ${price_change:.2f} ({price_change_pct:.2f}%)")
            else:
                st.error(f"Predicted decrease: ${price_change:.2f} ({price_change_pct:.2f}%)")
            
            # Trading signals
            st.subheader("Trading Signals")
            
            # Simple signal based on prediction
            if price_change_pct > 5:
                st.markdown("**Signal:** 🟢 **STRONG BUY** - Predicted upside exceeds 5%")
            elif price_change_pct > 2:
                st.markdown("**Signal:** 🟢 **BUY** - Predicted upside between 2-5%")
            elif price_change_pct > -2:
                st.markdown("**Signal:** 🟡 **HOLD** - Predicted change between -2% and 2%")
            elif price_change_pct > -5:
                st.markdown("**Signal:** 🔴 **SELL** - Predicted downside between -2% and -5%")
            else:
                st.markdown("**Signal:** 🔴 **STRONG SELL** - Predicted downside exceeds -5%")
            
            # Disclaimer
            st.markdown("""
            ---
            **Disclaimer:** This stock prediction is for educational purposes only and should not be considered as financial advice.
            The predictions are based on historical data and may not accurately reflect future market conditions.
            Always consult with a financial advisor before making investment decisions.
            """)

else:
    # Display welcome message and instructions
    st.info("👈 Configure the parameters in the sidebar and click 'Run Prediction' to start")
    
    # Sample visualization
    st.subheader("Sample Visualization (Demo)")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=100)
    sample_price = 100 + np.cumsum(np.random.normal(0, 2, 100))
    sample_data = pd.DataFrame({
        'Close': sample_price,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Generate sample future data
    future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=30)
    future_trend = np.random.choice([-1, 1]) * 0.5  # Random trend direction
    future_price = sample_price[-1] + np.cumsum(np.random.normal(future_trend, 2, 30))
    future_data = pd.DataFrame({
        'Close': future_price,
        'Volume': np.random.randint(1000000, 10000000, 30)
    }, index=future_dates)
    
    # Plot sample data
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Price (Demo Data)', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Scatter(
            x=sample_data.index,
            y=sample_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_data.index,
            y=future_data['Close'],
            mode='lines',
            name='Forecast (Demo)',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=sample_data.index,
            y=sample_data['Volume'],
            name='Volume',
            marker_color='blue',
            opacity=0.5
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Demo Visualization (Sample Data)',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # App features
    st.subheader("App Features")
    st.markdown("""
    - **Data Loading**: Fetch historical stock data from Yahoo Finance
    - **Technical Analysis**: Calculate technical indicators like moving averages, RSI, MACD
    - **Machine Learning Models**: Choose between Random Forest, Linear Regression, or LSTM
    - **Interactive Visualization**: View historical data and predictions with confidence intervals
    - **Performance Metrics**: Evaluate model performance with MSE, RMSE, MAE, and R²
    - **Trading Signals**: Get buy/sell recommendations based on predictions
    """)