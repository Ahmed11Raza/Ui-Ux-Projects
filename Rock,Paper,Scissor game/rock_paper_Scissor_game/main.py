import streamlit as st
import random

# Set page configuration
st.set_page_config(
    page_title="Rock Paper Scissors",
    page_icon="✂️",
    layout="centered"
)

# Initialize session state variables
if 'wins' not in st.session_state:
    st.session_state.wins = 0
if 'losses' not in st.session_state:
    st.session_state.losses = 0
if 'draws' not in st.session_state:
    st.session_state.draws = 0
if 'total_games' not in st.session_state:
    st.session_state.total_games = 0
if 'result_displayed' not in st.session_state:
    st.session_state.result_displayed = False

# Page header
st.title("Rock Paper Scissors Game")
st.markdown("---")

# Game instructions
with st.expander("How to Play", expanded=False):
    st.markdown("""
    ### Game Rules:
    - Rock crushes Scissors
    - Scissors cuts Paper
    - Paper covers Rock
    
    Make your choice by clicking one of the buttons below. The computer will randomly select its choice.
    The result will be displayed immediately.
    """)

# Define game functions
def determine_winner(user_choice, computer_choice):
    """Determine the winner of the game."""
    if user_choice == computer_choice:
        return "It's a draw!"
        
    winning_combinations = {
        ("Rock", "Scissors"),
        ("Paper", "Rock"),
        ("Scissors", "Paper")
    }
    
    if (user_choice, computer_choice) in winning_combinations:
        return "You win!"
    else:
        return "You lose!"

def play_game(user_choice):
    """Play a game with the given user choice."""
    # Generate computer choice
    choices = ["Rock", "Paper", "Scissors"]
    computer_choice = random.choice(choices)
    
    # Determine winner
    result = determine_winner(user_choice, computer_choice)
    
    # Update statistics
    st.session_state.total_games += 1
    if result == "You win!":
        st.session_state.wins += 1
    elif result == "You lose!":
        st.session_state.losses += 1
    else:
        st.session_state.draws += 1
    
    # Display result
    st.markdown("---")
    st.markdown("### Results:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Your choice:** {user_choice}")
    with col2:
        st.markdown(f"**Computer's choice:** {computer_choice}")
    
    # Apply color based on result
    if result == "You win!":
        result_color = "green"
    elif result == "You lose!":
        result_color = "red"
    else:
        result_color = "blue"
    
    st.markdown(
        f"<h3 style='text-align: center; color: {result_color};'>{result}</h3>",
        unsafe_allow_html=True
    )
    
    st.session_state.result_displayed = True

# Main game section
st.markdown("### Make Your Choice:")

# Create columns for buttons
col1, col2, col3 = st.columns(3)

# Button handling
choice_made = False
with col1:
    if st.button("🪨 Rock", use_container_width=True):
        play_game("Rock")
        choice_made = True

with col2:
    if st.button("📄 Paper", use_container_width=True):
        play_game("Paper")
        choice_made = True

with col3:
    if st.button("✂️ Scissors", use_container_width=True):
        play_game("Scissors")
        choice_made = True

# Statistics section
st.markdown("---")
st.markdown("### Game Statistics:")

# Display statistics in a grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Games", st.session_state.total_games)
with col2:
    st.metric("Wins", st.session_state.wins)
with col3:
    st.metric("Losses", st.session_state.losses)
with col4:
    st.metric("Draws", st.session_state.draws)

# Calculate and display win percentage if games have been played
if st.session_state.total_games > 0:
    win_percentage = (st.session_state.wins / st.session_state.total_games) * 100
    st.progress(win_percentage / 100)
    st.text(f"Win Rate: {win_percentage:.1f}%")

# Reset button
if st.button("Reset Statistics"):
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.draws = 0
    st.session_state.total_games = 0
    st.session_state.result_displayed = False
    st.rerun()