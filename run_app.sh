#!/bin/bash

# Car Classification System Launcher
echo "🚗 Starting Car Classification System..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements_streamlit.txt

# Launch Streamlit app
echo "🚀 Launching Streamlit application..."
echo "Your app will open in your browser at: http://localhost:8501"
echo "=================================="

streamlit run MVP/Main.py