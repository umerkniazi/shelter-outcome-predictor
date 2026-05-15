import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/shelter_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

st.title("Shelter Outcome Predictor")
st.write("Estimate whether an animal is likely to be adopted, transferred, or at risk.")

animal_type = st.selectbox("Animal Type", ["Dog", "Cat", "Bird", "Other", "Livestock"])
age_years = st.slider("Age (years)", 0.0, 20.0, 1.0, step=0.5)
sex_status = st.radio("Sex Status", ["Neutered / Spayed", "Intact", "Unknown"])
is_mix = st.checkbox("Mixed Breed")
has_name = st.checkbox("Has a Name")
month = st.slider("Month", 1, 12, 6)
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

if st.button("Predict"):
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

    is_neutered = int(sex_status == "Neutered / Spayed")
    is_intact = int(sex_status == "Intact")

    input_data = {
        'age_days': age_years * 365,
        'is_mix': int(is_mix),
        'is_neutered': is_neutered,
        'is_intact': is_intact,
        'has_name': int(has_name),
        'month': month,
        'day_of_week': day_map[day_of_week],
        'animal_type_Bird': int(animal_type == 'Bird'),
        'animal_type_Cat': int(animal_type == 'Cat'),
        'animal_type_Dog': int(animal_type == 'Dog'),
        'animal_type_Livestock': int(animal_type == 'Livestock'),
        'animal_type_Other': int(animal_type == 'Other'),
    }

    df_input = pd.DataFrame([input_data])[feature_columns]
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    proba_df = pd.DataFrame({'Outcome': model.classes_, 'Probability': proba}).sort_values('Probability', ascending=False)

    st.subheader(f"Predicted Outcome: {prediction}")
    st.bar_chart(proba_df.set_index('Outcome'))