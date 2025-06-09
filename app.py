import streamlit as st
import joblib
import numpy as np

#show dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
 
# st.title("Team Performance Dashboard")
 
# # Load dataset (make sure file path is correct)
# df = pd.read_csv("sprint_data_sample_300_with_ids.csv.crdownload")
 
# # Clean column names if needed
# df.columns = df.columns.str.strip()
 
# # Convert Delay column to Yes=1, No=0 for easier calculations
# df['Delay_flag'] = df['Delay'].map({'Yes': 1, 'No': 0})
 
# # Group by Team Name and aggregate
# team_stats = df.groupby('Team Name').agg(
#     total_sprints=('Sprint ID', 'count'),
#     delay_count=('Delay_flag', 'sum'),
#     delay_rate=('Delay_flag', 'mean'),
#     avg_planned_points=('Planned Story Points', 'mean'),
#     avg_completed_points=('Completed Story Points', 'mean')
# ).reset_index()
 
# # Sort teams by delay_rate descending (most delayed first)
# team_stats = team_stats.sort_values(by='delay_rate', ascending=False)
 
# # Show the table
# st.subheader("Team Delay Summary")
# st.dataframe(team_stats)
 
# col1, col2 = st.columns(2)
 
# with col1:
#     st.subheader("Delayed Sprints Distribution by Team")
#     fig1, ax1 = plt.subplots()
#     fig1.patch.set_facecolor("#5d5d5d")
#     ax1.pie(
#         team_stats['delay_count'],
#         labels=team_stats['Team Name'],
#         autopct='%1.1f%%',
#         colors=plt.cm.Pastel1.colors,
#         startangle=140
#     )
#     ax1.set_title("Delayed Sprints")
#     st.pyplot(fig1)
 
# with col2:
#     st.subheader("Delay Rate Distribution by Team")
#     fig2, ax2 = plt.subplots()
#     fig2.patch.set_facecolor("#5d5d5d")
#     ax2.pie(
#         team_stats['delay_rate'],
#         labels=team_stats['Team Name'],
#         autopct='%1.1f%%',
#         colors=plt.cm.Pastel2.colors,
#         startangle=140
#     )
#     ax2.set_title("Delay Rate")
#     st.pyplot(fig2)
    
 
# Load models and encoders
model = joblib.load("delay_category_model.pkl")
le_delay_cat = joblib.load("delay_category_label_encoder.pkl")
label_encoders = joblib.load("label_encoders.pkl")
 
st.title("PI Health : Project Delay Category Predictor")
 
with st.form("delay_form"):
    st.subheader("Enter Sprint Information")
    planned_points = st.number_input("Planned Story Points", min_value=0)
    completed_points = st.number_input("Completed Story Points", min_value=0)
    sprint_duration = st.number_input("Sprint Duration (days)", min_value=1)
    blockers_count = st.number_input("Blockers Count", min_value=0)
    team_size = st.number_input("Team Size", min_value=1)
    holidays = st.number_input("Holidays", min_value=0)
 
    # Dynamically get category options from encoders
    unavailability_options = label_encoders["Unavailability"].classes_
    prev_delay_options = label_encoders["Previous Sprint Delay"].classes_
 
    unavailability = st.selectbox("Team Unavailability", unavailability_options)
    prev_delay = st.selectbox("Previous Sprint Delay", prev_delay_options)
 
    submitted = st.form_submit_button("Predict Delay Category")
 
if submitted:
    # Encode categorical inputs
    unavailability_encoded = label_encoders["Unavailability"].transform([unavailability])[0]
    prev_delay_encoded = label_encoders["Previous Sprint Delay"].transform([prev_delay])[0]
 
    # Prepare input in correct order
    input_data = pd.DataFrame([{
        "Planned Story Points" : planned_points,
        "Completed Story Points" : completed_points,
        "Sprint Duration" : sprint_duration,
        "Blockers Count" : blockers_count,
        "Team Size" : team_size,
        "Holidays" : holidays,
        "Unavailability" : unavailability_encoded,
        "Previous Sprint Delay" : prev_delay_encoded
    }])
    # Predict encoded category
    pred_encoded = model.predict(input_data)[0]
 
    # Decode to human-readable category
    pred_category = le_delay_cat.inverse_transform([pred_encoded])[0]
 
    # Display result
    st.markdown(f"### Predicted Delay Category: **{pred_category}**")

    