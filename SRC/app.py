import pickle
import streamlit as st
import numpy as np 
import sqlite3
import pandas as pd
from sklearn.metrics import log_loss
from pathlib import Path

# Load the model once, to avoid reloading it on each input change
@st.cache_data
def load_model():
    with open(Path(__file__).parent / 'lr.pkl', 'rb') as file:
        return pickle.load(file)


@st.cache_data
def load_model1():
    with open(Path(__file__).parent / 'sc.pkl', 'rb') as file:
        return pickle.load(file)

lr = load_model()
sc = load_model1()
db_path = Path(__file__).parent / 'data2022.sqlite'
con = sqlite3.connect(db_path)

# List of currently active NBA teams
# List of currently active NBA teams
team_abbreviations = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}


# Title and description
st.title('🏀 NBA Win Predictor')
st.write("### Predict the winner based on game statistics")

# Team selection dropdowns
hometeam = st.selectbox('Select Home Team', options=list(team_abbreviations.values()))
awayteam = st.selectbox('Select Away Team', options=list(team_abbreviations.values()))

# Predict based on team selection
if st.button('Predict Based on Teams'):
    if hometeam and awayteam:
        if hometeam == awayteam:
            st.error("Home and Away teams cannot be the same. Please select different teams.")
        else:
            query = '''
            SELECT 
                t1.fg_pct_home,
                t2.fg_pct_away,
                t1.fg3_pct_home,
                t2.fg3_pct_away,
                t1.reb_home,
                t2.reb_away,
                t1.ast_home,
                t2.ast_away,
                t1.tov_home,
                t2.tov_away,
                t1.win_ratio_home,
                t2.win_ratio_away
            FROM
                (SELECT 
                    avg(fg_pct_home) AS fg_pct_home,
                    avg(fg3_pct_home) AS fg3_pct_home,
                    avg(reb_home) AS reb_home,
                    avg(ast_home) AS ast_home,
                    avg(tov_home) AS tov_home,
                    avg(CASE WHEN wl_home = 'W' THEN 1 ELSE 0 END) AS win_ratio_home
                FROM Game 
                WHERE team_abbreviation_home = ?
                AND game_date BETWEEN '2022-01-01' AND '2023-01-01') AS t1
            CROSS JOIN
                (SELECT 
                    avg(fg_pct_away) AS fg_pct_away,
                    avg(fg3_pct_away) AS fg3_pct_away,
                    avg(reb_away) AS reb_away,
                    avg(ast_away) AS ast_away,
                    avg(tov_away) AS tov_away,
                    avg(CASE WHEN wl_away = 'W' THEN 1 ELSE 0 END) AS win_ratio_away
                FROM Game 
                WHERE team_abbreviation_away = ? 
                AND game_date BETWEEN '2022-01-01' AND '2023-01-01') AS t2;
            '''
            data_test = pd.read_sql_query(query, con, params=(hometeam, awayteam))

            if not data_test.empty:
                # Prepare features
                features = data_test.values
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                features = sc.transform(features)

                # Predict the outcome
                prediction = lr.predict(features)[0]
                probs = lr.predict_proba(features)
            

                # Display result
                if prediction == 1:
                    i=probs[0][1]
                    st.success(f"🏆 {hometeam} is predicted to win!")
                    st.success(f"Probability of winning is :{i}")
                else:
                    j=probs[0][0]
                    st.success(f"🏆 {awayteam} is predicted to win!")
                    st.success(f"Probability of winning is :{j}")
            else:
                st.error("No data available for the selected teams.")
    else:
        st.warning("Please select both Home and Away teams.")

# Manual input form
st.write("### Or enter manually")

with st.form(key='manual_input_form'):
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input('Home Team Field Goals Successful', min_value=0.0, max_value=100.0, value=50.0)
        b = st.number_input('Home Team Field Goals Attempted', min_value=0.0, max_value=100.0, value=50.0)
        fg_pct_home = a * 1.0 / b

        e = st.number_input('Home Team 3-Point Field Goals Successful', min_value=0.0, max_value=100.0, value=35.0)
        f = st.number_input('Home Team 3-Point Field Goals Attempted', min_value=0.0, max_value=100.0, value=35.0)
        fg3_pct_home = e * 1.0 / f

        reb_home = st.number_input('Home Team Rebounds', min_value=0, value=40)
        ast_home = st.number_input('Home Team Assists', min_value=0, value=25)
        tov_home = st.number_input('Home Team Turnovers', min_value=0, value=15)

    with col2:
        c = st.number_input('Away Team Field Goals Successful', min_value=0.0, max_value=100.0, value=50.0)
        d = st.number_input('Away Team Field Goals Attempted', min_value=0.0, max_value=100.0, value=50.0)
        fg_pct_away = c * 1.0 / d

        g = st.number_input('Away Team 3-Point Field Goals Successful', min_value=0.0, max_value=100.0, value=35.0)
        h = st.number_input('Away Team 3-Point Field Goals Attempted', min_value=0.0, max_value=100.0, value=35.0)
        fg3_pct_away = g * 1.0 / h

        reb_away = st.number_input('Away Team Rebounds', min_value=0, value=40)
        ast_away = st.number_input('Away Team Assists', min_value=0, value=25)
        tov_away = st.number_input('Away Team Turnovers', min_value=0, value=15)

    submit_button = st.form_submit_button(label='Predict Based on Manual Inputs')
    
    if submit_button:
        if hometeam and awayteam and hometeam != awayteam:
            # Retrieve win ratios for the selected teams
            query_win_ratio = '''
            SELECT 
                avg(CASE WHEN wl_home = 'W' THEN 1 ELSE 0 END) AS win_ratio_home,
                avg(CASE WHEN wl_away = 'W' THEN 1 ELSE 0 END) AS win_ratio_away
            FROM Game 
            WHERE (team_abbreviation_home = ? AND game_date BETWEEN '2022-01-01' AND '2023-01-01')
            OR (team_abbreviation_away = ? AND game_date BETWEEN '2022-01-01' AND '2023-01-01');
            '''
            win_ratios = pd.read_sql_query(query_win_ratio, con, params=(hometeam, awayteam))

            if not win_ratios.empty:
                win_ratio_home = win_ratios['win_ratio_home'].iloc[0]
                win_ratio_away = win_ratios['win_ratio_away'].iloc[0]

                # Prepare features from manual inputs
                features_manual = np.array([[fg_pct_home, fg_pct_away, fg3_pct_home, fg3_pct_away, reb_home, reb_away, ast_home, ast_away, tov_home, tov_away, win_ratio_home, win_ratio_away]])
                features_manual = sc.transform(features_manual)

                # Predict the outcome
                prediction_manual = lr.predict(features_manual)[0]

                # Display result
                if prediction_manual == 1:
                    st.success(f"🏆 {hometeam} is predicted to win!")
                else:
                    st.success(f"🏆 {awayteam} is predicted to win!")
            else:
                st.error("Could not retrieve win ratios for the selected teams.")
        else:
            st.warning("Please select different teams for Home and Away.")
