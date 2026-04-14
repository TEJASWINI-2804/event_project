import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Smart Event Recommendation System")

# Inputs
dept = st.selectbox("Department", ["CSE", "ECE", "BBA"])
year = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
interest = st.selectbox("Interest", ["Technical", "Cultural", "Sports", "Workshops"])
time = st.selectbox("Preferred Time", ["Morning", "Afternoon", "Evening"])
frequency = st.selectbox("Participation", ["Rare", "Sometimes", "Often"])

# Simple encoding (same order as training)
mapping = {
    "CSE": 0, "ECE": 1, "BBA": 2,
    "1st Year": 0, "2nd Year": 1, "3rd Year": 2, "4th Year": 3,
    "Technical": 0, "Cultural": 1, "Sports": 2, "Workshops": 3,
    "Morning": 0, "Afternoon": 1, "Evening": 2,
    "Rare": 0, "Sometimes": 1, "Often": 2
}

if st.button("Recommend Event"):
    input_data = [[
        mapping[dept],
        mapping[year],
        mapping[interest],
        mapping[time],
        mapping[frequency]
    ]]

    prediction = model.predict(input_data)

    events = ["Hackathon", "Dance Fest", "Seminar", "Sports Meet"]

    st.success(f" Recommended Event: {events[prediction[0]]}")
    st.info("Based on your preferences and past student data")