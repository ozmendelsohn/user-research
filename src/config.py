"""Configuration settings for the application."""

# LLM Settings
LLM_MODEL = "gpt-4o-mini" 
MAX_TOKENS = 4096

# File paths
import os

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# Add these constants
STREAMLIT_STYLE = """
<style>
    .stButton button {
        width: 100%;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""" 

# Add LLM settings
LLM_SETTINGS = {
    "groq": {
        "model": "llama-3.1-70b-versatile",
        "temperature": 0.7
    },
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.7
    },
    "ollama": {
        "model": "llama3.2:1b",  # or whichever model you have pulled
        "temperature": 0.7
    }
}

# Default provider
DEFAULT_LLM_PROVIDER = "ollama" 