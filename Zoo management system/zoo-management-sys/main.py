import streamlit as st
import uuid
from datetime import datetime, date
import pandas as pd

class Animal:
    def __init__(self, name, species, age, habitat, diet):
        self.id = str(uuid.uuid4())
        self.name = name
        self.species = species
        self.age = age
        self.habitat = habitat
        self.diet = diet
        self.arrival_date = date.today()
        self.health_status = "Healthy"
        self.medical_history = []

    def update_health(self, status, notes=None):
        self.health_status = status
        if notes:
            self.medical_history.append({
                'date': datetime.now(),
                'notes': notes
            })

class Caretaker:
    def __init__(self, name, specialization, contact):
        self.id = str(uuid.uuid4())
        self.name = name
        self.specialization = specialization
        self.contact = contact
        self.assigned_animals = []

class Exhibit:
    def __init__(self, name, capacity, type_of_habitat):
        self.id = str(uuid.uuid4())
        self.name = name
        self.capacity = capacity
        self.type_of_habitat = type_of_habitat
        self.current_animals = []

class ZooManagementSystem:
    def __init__(self):
        self.animals = {}
        self.caretakers = {}
        self.exhibits = {}

    def add_animal(self, animal):
        self.animals[animal.id] = animal
        return animal.id

    def add_caretaker(self, caretaker):
        self.caretakers[caretaker.id] = caretaker
        return caretaker.id

    def add_exhibit(self, exhibit):
        self.exhibits[exhibit.id] = exhibit
        return exhibit.id

    def get_animals(self):
        return list(self.animals.values())

    def get_caretakers(self):
        return list(self.caretakers.values())

    def get_exhibits(self):
        return list(self.exhibits.values())

def main():
    # Initialize Zoo Management System
    if 'zoo' not in st.session_state:
        st.session_state.zoo = ZooManagementSystem()
    
    # Main title
    st.title("🦁 Zoo Management System")
    
    # Sidebar navigation
    menu = st.sidebar.selectbox("Menu", [
        "Dashboard", 
        "Animal Management", 
        "Caretaker Management", 
        "Exhibit Management",
        "Animal Health Tracking"
    ])
    
    zoo = st.session_state.zoo
    
    # Dashboard
    if menu == "Dashboard":
        st.header("Zoo Dashboard")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Animals", len(zoo.get_animals()))
        with col2:
            st.metric("Total Caretakers", len(zoo.get_caretakers()))
        with col3:
            st.metric("Total Exhibits", len(zoo.get_exhibits()))
        
        # Recent Animals
        st.subheader("Recent Animals")
        if zoo.get_animals():
            df = pd.DataFrame([
                {
                    'Name': animal.name, 
                    'Species': animal.species, 
                    'Age': animal.age, 
                    'Habitat': animal.habitat
                } for animal in zoo.get_animals()
            ])
            st.dataframe(df)
    
    # Animal Management
    elif menu == "Animal Management":
        st.header("Animal Management")
        
        # Add Animal Section
        st.subheader("Add New Animal")
        with st.form("add_animal_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Animal Name")
                species = st.text_input("Species")
            with col2:
                age = st.number_input("Age", min_value=0)
                habitat = st.selectbox("Habitat", [
                    "Savanna", "Tropical Forest", "Arctic", 
                    "Desert", "Aquarium", "Aviary"
                ])
            
            diet = st.text_input("Diet")
            submit = st.form_submit_button("Add Animal")
            
            if submit:
                if name and species and age and habitat and diet:
                    new_animal = Animal(name, species, age, habitat, diet)
                    zoo.add_animal(new_animal)
                    st.success(f"Animal {name} added successfully!")
                else:
                    st.error("Please fill in all fields")
        
        # View Animals Section
        st.subheader("Current Animals")
        if zoo.get_animals():
            df = pd.DataFrame([
                {
                    'Name': animal.name, 
                    'Species': animal.species, 
                    'Age': animal.age, 
                    'Habitat': animal.habitat,
                    'Diet': animal.diet
                } for animal in zoo.get_animals()
            ])
            st.dataframe(df)
        else:
            st.info("No animals in the zoo yet.")
    
    # Caretaker Management
    elif menu == "Caretaker Management":
        st.header("Caretaker Management")
        
        # Add Caretaker Section
        st.subheader("Add New Caretaker")
        with st.form("add_caretaker_form"):
            name = st.text_input("Caretaker Name")
            specialization = st.selectbox("Specialization", [
                "Mammals", "Reptiles", "Birds", 
                "Aquatic", "Veterinary Care"
            ])
            contact = st.text_input("Contact Number")
            submit = st.form_submit_button("Add Caretaker")
            
            if submit:
                if name and specialization and contact:
                    new_caretaker = Caretaker(name, specialization, contact)
                    zoo.add_caretaker(new_caretaker)
                    st.success(f"Caretaker {name} added successfully!")
                else:
                    st.error("Please fill in all fields")
        
        # View Caretakers Section
        st.subheader("Current Caretakers")
        if zoo.get_caretakers():
            df = pd.DataFrame([
                {
                    'Name': caretaker.name, 
                    'Specialization': caretaker.specialization, 
                    'Contact': caretaker.contact
                } for caretaker in zoo.get_caretakers()
            ])
            st.dataframe(df)
        else:
            st.info("No caretakers added yet.")
    
    # Exhibit Management
    elif menu == "Exhibit Management":
        st.header("Exhibit Management")
        
        # Add Exhibit Section
        st.subheader("Add New Exhibit")
        with st.form("add_exhibit_form"):
            name = st.text_input("Exhibit Name")
            capacity = st.number_input("Capacity", min_value=1)
            habitat_type = st.selectbox("Habitat Type", [
                "Savanna", "Tropical Forest", "Arctic", 
                "Desert", "Aquarium", "Aviary"
            ])
            submit = st.form_submit_button("Add Exhibit")
            
            if submit:
                if name and capacity and habitat_type:
                    new_exhibit = Exhibit(name, capacity, habitat_type)
                    zoo.add_exhibit(new_exhibit)
                    st.success(f"Exhibit {name} added successfully!")
                else:
                    st.error("Please fill in all fields")
        
        # View Exhibits Section
        st.subheader("Current Exhibits")
        if zoo.get_exhibits():
            df = pd.DataFrame([
                {
                    'Name': exhibit.name, 
                    'Capacity': exhibit.capacity, 
                    'Habitat Type': exhibit.type_of_habitat
                } for exhibit in zoo.get_exhibits()
            ])
            st.dataframe(df)
        else:
            st.info("No exhibits added yet.")
    
    # Animal Health Tracking
    elif menu == "Animal Health Tracking":
        st.header("Animal Health Tracking")
        
        # Select Animal
        animals = zoo.get_animals()
        if animals:
            selected_animal = st.selectbox(
                "Select Animal", 
                [animal.name for animal in animals]
            )
            
            # Find the selected animal
            current_animal = next(
                (animal for animal in animals if animal.name == selected_animal), 
                None
            )
            
            if current_animal:
                # Display current health status
                st.subheader(f"Health Status: {current_animal.health_status}")
                
                # Update Health Status Form
                with st.form("health_update_form"):
                    new_status = st.selectbox("Update Health Status", [
                        "Healthy", "Sick", "Recovery", "Quarantine"
                    ])
                    medical_notes = st.text_area("Medical Notes")
                    submit = st.form_submit_button("Update Health Status")
                    
                    if submit:
                        current_animal.update_health(new_status, medical_notes)
                        st.success("Health status updated successfully!")
                
                # Medical History
                if current_animal.medical_history:
                    st.subheader("Medical History")
                    for entry in current_animal.medical_history:
                        st.write(f"Date: {entry['date']}")
                        st.write(f"Notes: {entry['notes']}")
        else:
            st.info("No animals to track health. Please add animals first.")

if __name__ == "__main__":
    main()