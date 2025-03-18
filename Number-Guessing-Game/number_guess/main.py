import streamlit as st
import random
import time
from typing import Tuple, Optional


class NumberGuessingGame:
    """
    A class to manage the number guessing game logic.
    """
    
    def __init__(self, min_value: int = 1, max_value: int = 100, max_attempts: int = 10):
        """
        Initialize the game with customizable parameters.
        
        Args:
            min_value: The minimum value for the random number
            max_value: The maximum value for the random number
            max_attempts: Maximum number of attempts allowed
        """
        self.min_value = min_value
        self.max_value = max_value
        self.max_attempts = max_attempts
        self.target_number = None
        self.attempts_left = max_attempts
        self.game_over = False
        self.won = False
    
    def start_new_game(self) -> None:
        """Start a new game by generating a random number and resetting attempts."""
        self.target_number = random.randint(self.min_value, self.max_value)
        self.attempts_left = self.max_attempts
        self.game_over = False
        self.won = False
    
    def check_guess(self, guess: int) -> Tuple[str, bool]:
        """
        Check if the guess is correct and return appropriate feedback.
        
        Args:
            guess: The player's guessed number
            
        Returns:
            Tuple containing feedback message and boolean indicating if game continues
        """
        if self.game_over:
            return "Game is already over. Please start a new game.", False
        
        self.attempts_left -= 1
        
        if guess == self.target_number:
            self.won = True
            self.game_over = True
            return f"Congratulations! You've guessed the correct number: {self.target_number}!", False
        
        if self.attempts_left <= 0:
            self.game_over = True
            return f"Game over! You've run out of attempts. The number was {self.target_number}.", False
        
        hint = "higher" if guess < self.target_number else "lower"
        return f"Try a {hint} number. You have {self.attempts_left} attempts remaining.", True


class StreamlitInterface:
    """
    A class to manage the Streamlit interface for the game.
    """
    
    def __init__(self):
        """Initialize the interface and configure the page."""
        st.set_page_config(
            page_title="Number Guessing Game",
            page_icon="🎮",
            layout="centered"
        )
        
        # Initialize session state variables if they don't exist
        if 'game' not in st.session_state:
            st.session_state.game = NumberGuessingGame()
            st.session_state.game.start_new_game()
        
        if 'game_history' not in st.session_state:
            st.session_state.game_history = []
        
        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False
    
    def render_header(self) -> None:
        """Render the application header."""
        st.title("Number Guessing Game")
        st.markdown("""
        Try to guess the secret number within the given number of attempts.
        The application will provide hints after each guess.
        """)
        st.divider()
    
    def render_settings(self) -> None:
        """Render the game settings section."""
        if st.button("Game Settings" if not st.session_state.show_settings else "Hide Settings"):
            st.session_state.show_settings = not st.session_state.show_settings
        
        if st.session_state.show_settings:
            with st.form("settings_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    min_value = st.number_input(
                        "Minimum Value",
                        value=1,
                        min_value=1,
                        step=1
                    )
                    
                    max_value = st.number_input(
                        "Maximum Value",
                        value=100,
                        min_value=min_value + 10,
                        step=10
                    )
                
                with col2:
                    max_attempts = st.number_input(
                        "Maximum Attempts",
                        value=10,
                        min_value=1,
                        max_value=20,
                        step=1
                    )
                
                if st.form_submit_button("Apply Settings & Start New Game"):
                    st.session_state.game = NumberGuessingGame(
                        min_value=min_value,
                        max_value=max_value,
                        max_attempts=max_attempts
                    )
                    st.session_state.game.start_new_game()
                    st.session_state.game_history = []
                    st.success("New game started with updated settings!")
                    time.sleep(1)
                    st.rerun()
    
    def render_game_area(self) -> None:
        """Render the main game area with input and feedback."""
        game = st.session_state.game
        
        # Game status indicators
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Attempts Remaining", game.attempts_left)
        with col2:
            st.metric("Number Range", f"{game.min_value} - {game.max_value}")
        
        # Game input
        if not game.game_over:
            with st.form("guess_form"):
                guess = st.number_input(
                    "Enter your guess:",
                    min_value=game.min_value,
                    max_value=game.max_value,
                    step=1
                )
                
                submitted = st.form_submit_button("Submit Guess")
                
                if submitted:
                    feedback, _ = game.check_guess(guess)
                    st.session_state.game_history.append({
                        "guess": guess,
                        "feedback": feedback
                    })
                    st.rerun()
        
        # Game over state
        else:
            result_message = "🎉 You Won! 🎉" if game.won else "Game Over!"
            result_color = "success" if game.won else "error"
            
            st.markdown(f"<h2 style='text-align: center;'>{result_message}</h2>", unsafe_allow_html=True)
            
            if st.button("Start New Game", type="primary"):
                game.start_new_game()
                st.session_state.game_history = []
                st.rerun()
    
    def render_history(self) -> None:
        """Render the game history section."""
        if st.session_state.game_history:
            st.subheader("Guess History")
            
            for i, entry in enumerate(st.session_state.game_history, 1):
                with st.container():
                    cols = st.columns([1, 6])
                    with cols[0]:
                        st.markdown(f"**#{i}**")
                    with cols[1]:
                        st.markdown(f"Guess: **{entry['guess']}** - {entry['feedback']}")
    
    def render_app(self) -> None:
        """Render the complete application."""
        self.render_header()
        self.render_settings()
        
        st.divider()
        self.render_game_area()
        
        st.divider()
        self.render_history()


# Main application entry point
def main():
    """Main function to run the Streamlit application."""
    app = StreamlitInterface()
    app.render_app()


if __name__ == "__main__":
    main()