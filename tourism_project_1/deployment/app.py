import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="kjdeka/tourism_package_prediction_model", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts likelihood of purchasing the Wellness Tourism Package.
Please enter the required data below to get a prediction.
""")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=180.0, value=10.0, step=0.5)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=3, step=1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=2, step=1)
passport = st.selectbox("Passport", [0, 1])  # 0: No, 1: Yes
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])  # 0: No, 1: Yes
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Purchase" if prediction == 1 else "Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
