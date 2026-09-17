
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


df = pd.read_csv("hour.csv")

drop_columns = ["cnt", "casual", "registered", "instant", "dteday"]

X = df.drop(columns=drop_columns)
y = df["cnt"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)


st.title("🚲 Bike Sharing Demand Prediction")

st.write("Enter the bike sharing information:")


season = st.number_input("Season", 1, 4, 2)
yr = st.number_input("Year", 0, 1, 1)
mnth = st.number_input("Month", 1, 12, 6)
hr = st.number_input("Hour", 0, 23, 18)
holiday = st.number_input("Holiday", 0, 1, 0)
weekday = st.number_input("Weekday", 0, 6, 2)
workingday = st.number_input("Working Day", 0, 1, 1)
weathersit = st.number_input("Weather Situation", 1, 4, 1)

temp = st.number_input("Temperature", 0.0, 1.0, 0.60)
atemp = st.number_input("Feeling Temperature", 0.0, 1.0, 0.58)
hum = st.number_input("Humidity", 0.0, 1.0, 0.55)
windspeed = st.number_input("Windspeed", 0.0, 1.0, 0.20)


if st.button("Predict"):

    new_data = pd.DataFrame({
        "season": [season],
        "yr": [yr],
        "mnth": [mnth],
        "hr": [hr],
        "holiday": [holiday],
        "weekday": [weekday],
        "workingday": [workingday],
        "weathersit": [weathersit],
        "temp": [temp],
        "atemp": [atemp],
        "hum": [hum],
        "windspeed": [windspeed]
    })

    prediction = model.predict(new_data)

    st.success(f"Predicted Bike Rentals: {prediction[0]:.0f}")

