import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

class AirportManagementSystem:
    def __init__(self):
        # Initialize database connection
        self.conn = sqlite3.connect('airport_management.db')
        self.create_tables()

    def create_tables(self):
        """Create necessary database tables"""
        cursor = self.conn.cursor()
        
        # Flights table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS flights (
            flight_id TEXT PRIMARY KEY,
            airline TEXT,
            origin TEXT,
            destination TEXT,
            departure_time DATETIME,
            arrival_time DATETIME,
            status TEXT,
            gate TEXT
        )
        ''')
        
        # Passengers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS passengers (
            passenger_id TEXT PRIMARY KEY,
            name TEXT,
            flight_id TEXT,
            seat_number TEXT,
            boarding_time DATETIME,
            FOREIGN KEY(flight_id) REFERENCES flights(flight_id)
        )
        ''')
        
        # Gates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS gates (
            gate_number TEXT PRIMARY KEY,
            current_flight TEXT,
            availability TEXT
        )
        ''')
        
        self.conn.commit()

    def add_flight(self, flight_id, airline, origin, destination, departure_time, arrival_time, status, gate):
        """Add a new flight to the database"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO flights 
        (flight_id, airline, origin, destination, departure_time, arrival_time, status, gate) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (flight_id, airline, origin, destination, departure_time, arrival_time, status, gate))
        self.conn.commit()

    def add_passenger(self, passenger_id, name, flight_id, seat_number):
        """Add a new passenger to a flight"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO passengers 
        (passenger_id, name, flight_id, seat_number, boarding_time) 
        VALUES (?, ?, ?, ?, ?)
        ''', (passenger_id, name, flight_id, seat_number, datetime.now()))
        self.conn.commit()

    def update_flight_status(self, flight_id, status):
        """Update flight status"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE flights SET status = ? WHERE flight_id = ?', (status, flight_id))
        self.conn.commit()

    def get_flights(self):
        """Retrieve all flights"""
        return pd.read_sql_query("SELECT * FROM flights", self.conn)

    def get_passengers(self, flight_id=None):
        """Retrieve passengers, optionally filtered by flight"""
        if flight_id:
            return pd.read_sql_query("SELECT * FROM passengers WHERE flight_id = ?", self.conn, params=(flight_id,))
        return pd.read_sql_query("SELECT * FROM passengers", self.conn)

def main():
    st.title("🛫 Airport Management System")
    
    # Initialize airport management system
    ams = AirportManagementSystem()

    # Sidebar navigation
    menu = st.sidebar.selectbox("Menu", [
        "Dashboard", 
        "Add Flight", 
        "Add Passenger", 
        "Flight Status", 
        "Passenger List"
    ])

    if menu == "Dashboard":
        st.header("Airport Dashboard")
        flights_df = ams.get_flights()
        
        # Flight Summary
        st.subheader("Flight Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Flights", len(flights_df))
        
        with col2:
            st.metric("On-Time Flights", len(flights_df[flights_df['status'] == 'On Time']))
        
        with col3:
            st.metric("Delayed Flights", len(flights_df[flights_df['status'] == 'Delayed']))
        
        st.dataframe(flights_df)

    elif menu == "Add Flight":
        st.header("Add New Flight")
        with st.form("flight_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                flight_id = st.text_input("Flight ID")
                airline = st.text_input("Airline")
                origin = st.text_input("Origin")
                destination = st.text_input("Destination")
            
            with col2:
                departure_time = st.datetime_input("Departure Time")
                arrival_time = st.datetime_input("Arrival Time")
                status = st.selectbox("Status", ["Scheduled", "On Time", "Delayed", "Cancelled"])
                gate = st.text_input("Gate")
            
            submit_button = st.form_submit_button("Add Flight")
            
            if submit_button:
                ams.add_flight(flight_id, airline, origin, destination, 
                               departure_time, arrival_time, status, gate)
                st.success(f"Flight {flight_id} added successfully!")

    elif menu == "Add Passenger":
        st.header("Add Passenger to Flight")
        with st.form("passenger_form"):
            passenger_id = st.text_input("Passenger ID")
            name = st.text_input("Passenger Name")
            flight_id = st.text_input("Flight ID")
            seat_number = st.text_input("Seat Number")
            
            submit_button = st.form_submit_button("Add Passenger")
            
            if submit_button:
                ams.add_passenger(passenger_id, name, flight_id, seat_number)
                st.success(f"Passenger {name} added to Flight {flight_id}")

    elif menu == "Flight Status":
        st.header("Update Flight Status")
        flights = ams.get_flights()
        flight_ids = flights['flight_id'].tolist()
        
        selected_flight = st.selectbox("Select Flight", flight_ids)
        new_status = st.selectbox("Update Status", ["Scheduled", "On Time", "Delayed", "Cancelled"])
        
        if st.button("Update Status"):
            ams.update_flight_status(selected_flight, new_status)
            st.success(f"Flight {selected_flight} status updated to {new_status}")

    elif menu == "Passenger List":
        st.header("Passenger List")
        selected_flight = st.selectbox("Select Flight", ams.get_flights()['flight_id'].tolist())
        
        passengers_df = ams.get_passengers(selected_flight)
        st.dataframe(passengers_df)

if __name__ == "__main__":
    main()