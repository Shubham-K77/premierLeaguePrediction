#Import Streamlit And Pickle
import streamlit as st # type: ignore
import pickle
import pandas as pd
import plotly.graph_objects as go # type: ignore
#Constants:
display_labels = [
        'Preferred Team:',
        'Rival Team:',
        'Match Venue:',
        'Playing Formation:',
        'Match Referee:',
        'Matchday:',
]
preferred_team_mapping = {
'Arsenal': 0,
 'Aston Villa': 1,
 'Brentford': 2,
 'Brighton and Hove Albion': 3,
 'Burnley': 4,
 'Chelsea': 5,
 'Crystal Palace': 6,
 'Everton': 7,
 'Fulham': 8,
 'Leeds United': 9,
 'Leicester City': 10,
 'Liverpool': 11,
 'Manchester City': 12,
 'Manchester United': 13,
 'Newcastle United': 14,
 'Norwich City': 15,
 'Sheffield United': 16,
 'Southampton': 17,
 'Tottenham Hotspur': 18,
 'Watford': 19,
 'West Bromwich Albion': 20,
 'West Ham United': 21,
 'Wolverhampton Wanderers': 22
}
rival_team_mapping = {
 'Arsenal': 0,
 'Aston Villa': 1,
 'Brentford': 2,
 'Brighton': 3,
 'Burnley': 4,
 'Chelsea': 5,
 'Crystal Palace': 6,
 'Everton': 7,
 'Fulham': 8,
 'Leeds United': 9,
 'Leicester City': 10,
 'Liverpool': 11,
 'Manchester City': 12,
 'Manchester Utd': 13,
 'Newcastle Utd': 14,
 'Norwich City': 15,
 'Sheffield Utd': 16,
 'Southampton': 17,
 'Tottenham': 18,
 'Watford': 19,
 'West Brom': 20,
 'West Ham': 21,
 'Wolves': 22
}
venue_mapping = {
    "Home 🟢": 1,
    "Away 🔴": 0,
}
formation_mapping = {
 '3-4-1-2': 0,
 '3-4-3': 1,
 '3-4-3': 2,
 '3-5-1-1': 3,
 '3-5-2': 4,
 '4-1-4-1': 5,
 '4-2-2-2': 6,
 '4-2-3-1': 7,
 '4-2-3-1': 8,
 '4-3-2-1': 9,
 '4-3-3': 10,
 '4-3-3': 11,
 '4-4-1-1': 12,
 '4-4-2': 13,
 '4-4-2': 14,
 '4-5-1': 15
}
referee_mapping = {
 'Andre Marriner': 0,
 'Andy Madley': 1,
 'Anthony Taylor': 2,
 'Chris Kavanagh': 3,
 'Craig Pawson': 4,
 'Darren England': 5,
 'David Coote': 6,
 'Graham Scott': 7,
 'Jarred Gillett': 8,
 'John Brooks': 9,
 'Jonathan Moss': 10,
 'Kevin Friend': 11,
 'Lee Mason': 12,
 'Martin Atkinson': 13,
 'Michael Oliver': 14,
 'Michael Salisbury': 15,
 'Mike Dean': 16,
 'Paul Tierney': 17,
 'Peter Bankes': 18,
 'Robert Jones': 19,
 'Simon Hooper': 20,
 'Stuart Attwell': 21,
 'Tony Harrington': 22
}
matchday_mapping = {
    "Matchday-1": 1,
    "Matchday-2": 2,
    "Matchday-3": 3,
    "Matchday-4": 4,
    "Matchday-5": 5,
    "Matchday-6": 6,
    "Matchday-7": 7,
    "Matchday-8": 8,
    "Matchday-9": 9,
    "Matchday-10": 10,
    "Matchday-11": 11,
    "Matchday-12": 12,
    "Matchday-13": 13,
    "Matchday-14": 14,
    "Matchday-15": 15,
    "Matchday-16": 16,
    "Matchday-17": 17,
    "Matchday-18": 18,
    "Matchday-19": 19,
    "Matchday-20": 20,
    "Matchday-21": 21,
    "Matchday-22": 22,
    "Matchday-23": 23,
    "Matchday-24": 24,
    "Matchday-25": 25,
    "Matchday-26": 26,
    "Matchday-27": 27,
    "Matchday-28": 28,
    "Matchday-29": 29,
    "Matchday-30": 30,
    "Matchday-31": 31,
    "Matchday-32": 32,
    "Matchday-33": 33,
    "Matchday-34": 34,
    "Matchday-35": 35,
    "Matchday-36": 36,
    "Matchday-37": 37,
    "Matchday-38": 38,
}
cont_cols = ['gf_rolling', 'ga_rolling', 'xg_rolling', 'xga_rolling', 'poss_rolling', 'attendance_rolling', 'sh_rolling', 'sot_rolling', 'dist_rolling', 'fk_rolling', 'pk_rolling', 'pkatt_rolling']
#Function To Get the Dataset:
def get_dataset():
    data = pd.read_csv('dataset/historical_data.csv')
    return data
#Function To Provide The Rolling Stats:
def get_rolling_stats(data, team, featureName):
    #Get the data for both the team and opponent according to latest data
    team_data = data[data['team'] == team].sort_values('date', ascending=False)
    rolling_stats = {}
    for col in featureName:
        rolling_stats[f'{col}_rolling'] = team_data[col].rolling(3, closed='left').mean().iloc[-1]
    return pd.DataFrame([rolling_stats])
#Function To Make Prediction:
def make_prediction(data):
    #SubHeader
    st.subheader("Match Outcome Random Forest Predictor:")
    #Model
    model = pickle.load(open("model/model.pkl", "rb"))
    result_encoder = pickle.load(open("model/resultEncoder.pkl", "rb"))
    #Rearrange to match the exact training order
    data = data[['round','venue','opponent','formation','referee','team','gf_rolling','ga_rolling','xg_rolling','xga_rolling','poss_rolling','attendance_rolling','sh_rolling','sot_rolling','dist_rolling','fk_rolling','pk_rolling','pkatt_rolling']]
    prediction = model.predict(data)
    prediction = result_encoder.inverse_transform(prediction)
    if prediction[0] == 'W':
        st.write("<span class='win'>Win</span>", unsafe_allow_html=True)
    elif prediction[0] == 'L':
        st.write("<span class='loss'>Loss</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='draw'>Draw</span>", unsafe_allow_html=True)
    #Prediction Probabilities
    probabilities = model.predict_proba(data)
    class_labels = result_encoder.classes_
    proba_df = pd.DataFrame(probabilities, columns=class_labels)
    st.write("The probabilities of the predictions:")
    st.dataframe(proba_df, hide_index=True)
    #Disclaimer
    st.write("Predictions are experimental and may not reflect current season performance.")
#Function To Create Sidebar:
def create_sidebar():
    st.sidebar.header("Match Information")
    #Displaying The Labels And The SelectBox
    for i in display_labels:
        if(i == "Preferred Team:"):
            st.session_state["team"] = st.sidebar.selectbox(f"{i}", list(preferred_team_mapping.keys()), key="team_select")
        elif(i == "Rival Team:"):
            available_opponents = [team for team in rival_team_mapping.keys() if team != st.session_state.get("team")]
            st.session_state["opponent"] = st.sidebar.selectbox(f"{i}", available_opponents, key="opponent_select")
        elif(i == "Match Venue:"):
            st.session_state["venue"] = st.sidebar.selectbox(f"{i}", list(venue_mapping.keys()), key="venue_select")
        elif(i == "Playing Formation:"):
            st.session_state["formation"] = st.sidebar.selectbox(f"{i}", list(formation_mapping.keys()), key="formation_select")
        elif(i == "Match Referee:"):
            st.session_state["referee"] = st.sidebar.selectbox(f"{i}", list(referee_mapping.keys()), key="referee_select")
        elif(i == "Matchday:"):
            st.session_state["round"] = st.sidebar.selectbox(f"{i}", list(matchday_mapping.keys()), key="round_select")
    #Getting The Additional Needed Values
    rolling_stats = get_rolling_stats(get_dataset(), preferred_team_mapping[st.session_state['team']], ['gf','ga','xg','xga','poss','attendance','sh','sot','dist','fk','pk','pkatt'])
    #Creating The Dataset:
    selected_inputs = pd.DataFrame([{
        "team": preferred_team_mapping[st.session_state["team"]],
        "opponent": rival_team_mapping[st.session_state["opponent"]],
        "venue": venue_mapping[st.session_state["venue"]],
        "formation": formation_mapping[st.session_state["formation"]],
        "referee": referee_mapping[st.session_state["referee"]],
        "round": matchday_mapping[st.session_state["round"]],
    }])
    selected_inputs = pd.concat([selected_inputs, rolling_stats], axis=1)
    #Scaling the Rolling_stats
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        selected_inputs[cont_cols] = scaler.transform(selected_inputs[cont_cols])
    #Printing The Dataset:
    return(selected_inputs)
#Function To Create RadarChart:
def get_radar_chart(value):
    fig = go.Figure(data=go.Scatterpolar(
        r=value.values[0],
        theta=[i for i in value.columns],
        fill='toself',
        fillcolor='rgba(34, 197, 94, 0.15)',  # Transparent blue fill
        line=dict(color='#10b981')  # Dark blue outline
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color='#e5e7eb', tickfont=dict(color='#10b981')),
            angularaxis=dict(color='#e5e7eb', tickfont=dict(color='#f3f4f6'))
        ),
        font=dict(color='#10b981'),
        showlegend=False
    )
    return fig
#Main Function:
def main():
    st.set_page_config(
        page_title = "Premier League Match Predictor",
        page_icon = "⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    #Import Css
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    #Sidebar
    selected_inputs = create_sidebar()
    #Main Container
    with st.container():
       st.title("Premier League Match Predictor")  #h1 header element
       st.write("Predict Premier League match outcomes using real data and machine learning. Enter team stats and see data-driven predictions, powered by rolling averages and pre-match features. Perfect for football fans and data enthusiasts!") #P element
    #Columns
    cols1, cols2 = st.columns([3, 2])
    with cols1:
        radar_chart = get_radar_chart(selected_inputs[cont_cols])
        st.plotly_chart(radar_chart)
        st.write("Rolling stats are calculated from each team's last 3 matches, using data from the 2021–2022 Premier League seasons. This app is for entertainment and analysis only, it does not promote betting and is not liable for any betting decisions.")
    with cols2:
        make_prediction(selected_inputs)
if __name__ == "__main__":
    main()

