"""
Main application entry point for the RAG-based code review tool.

This script initializes the Streamlit application and renders
the main UI components.
"""

import streamlit as st
from config_manager import ConfigManager


def main():
    """
    Initialize and run the main application.
    
    This function creates a ConfigManager instance and uses it to
    render the sidebar and main content of the application.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Peer Code Review with RAG",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize configuration manager
    config_manager = ConfigManager()

    # Render sidebar
    config_manager.render_sidebar()

    # Render main content
    config_manager.render_main_content()


if __name__ == "__main__":
    main()