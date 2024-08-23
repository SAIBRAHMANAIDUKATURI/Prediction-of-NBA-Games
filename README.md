# NBA Game Outcome Prediction

This project is a Streamlit application designed to predict NBA game outcomes using logistic regression. The model leverages various game statistics, such as field goal percentages, rebounds, assists, and turnovers, to determine the likelihood of a home team winning.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Features

- Predict NBA game outcomes based on various statistical inputs.
- Allows users to manually input game features for prediction.
- Connects to a SQLite database to retrieve historical data.
- Supports dropdown selection of currently active NBA teams.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SAIBRAHMANAIDUKATURI/Prediction-of-NBA-Games.git
   cd Prediction-of-NBA-Games

2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate 

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt

## Usage
1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py

2. **Interact with the app:**
    Use the dropdown to select teams.
    Enter game statistics manually or use historical data by giving just the team names
    Predict the outcome and view results.

## Dependencies

- Python 3.9+
- Streamlit
- SQLite3
- Scikit-learn
- Pandas
- Pickle


