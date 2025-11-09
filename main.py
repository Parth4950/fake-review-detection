import streamlit as st
import os
from home_page import home_page
from training import training
from testing import test_model

# Page configuration
st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 1.8rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Radio button styling */
    [data-testid="stSidebar"] label {
        color: white !important;
        font-size: 1.1rem;
        padding: 0.5rem;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Fake Review Detection")
st.sidebar.markdown("---")

# Add a navbar with buttons
nav = st.sidebar.radio(
    "Navigate",
    ["🛍️ Scrape Reviews", "🎓 Training", "🧪 Testing"],
    label_visibility='visible'
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color: white; padding: 1rem;'>
<h4>📖 About</h4>
<p>This system uses machine learning to detect fake product reviews using:</p>
<ul>
<li>TF-IDF Vectorization</li>
<li>Sentiment Analysis</li>
<li>Multiple ML Models</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Show content based on the selected button in the navbar
if nav == "🛍️ Scrape Reviews":
    home_page()
elif nav == "🎓 Training":
    training()
elif nav == "🧪 Testing":
    test_model()


