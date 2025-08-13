import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --------------------------
# Load model and scaler
# --------------------------
import joblib

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

st.title("Employee Attrition Prediction App")
st.write("Predict whether an employee will leave the company based on given features.")

# --------------------------
# Define categorical mappings (must match training)
# --------------------------
business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
department_map = {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2}
education_field_map = {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2, 'Medical': 3, 'Other': 4, 'Technical Degree': 5}
gender_map = {'Female': 0, 'Male': 1}
job_role_map = {
    'Healthcare Representative': 0, 'Human Resources': 1, 'Laboratory Technician': 2,
    'Manager': 3, 'Manufacturing Director': 4, 'Research Director': 5,
    'Research Scientist': 6, 'Sales Executive': 7, 'Sales Representative': 8
}
marital_status_map = {'Divorced': 0, 'Married': 1, 'Single': 2}
overtime_map = {'No': 0, 'Yes': 1}

# --------------------------
# Create Streamlit inputs
# --------------------------
age = st.number_input("Age", min_value=18, max_value=70, value=30)
business_travel = st.selectbox("Business Travel", list(business_travel_map.keys()))
daily_rate = st.number_input("Daily Rate", min_value=0, value=500)
department = st.selectbox("Department", list(department_map.keys()))
distance_from_home = st.number_input("Distance From Home", min_value=0, value=5)
education = st.number_input("Education (1-5)", min_value=1, max_value=5, value=3)
education_field = st.selectbox("Education Field", list(education_field_map.keys()))
environment_satisfaction = st.number_input("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
gender = st.selectbox("Gender", list(gender_map.keys()))
hourly_rate = st.number_input("Hourly Rate", min_value=0, value=60)
job_involvement = st.number_input("Job Involvement (1-4)", min_value=1, max_value=4, value=3)
job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
job_role = st.selectbox("Job Role", list(job_role_map.keys()))
job_satisfaction = st.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
monthly_rate = st.number_input("Monthly Rate", min_value=0, value=20000)
num_companies_worked = st.number_input("Num Companies Worked", min_value=0, value=2)
overtime = st.selectbox("OverTime", list(overtime_map.keys()))
percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, value=15)
performance_rating = st.number_input("Performance Rating (1-4)", min_value=1, max_value=4, value=3)
relationship_satisfaction = st.number_input("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
stock_option_level = st.number_input("Stock Option Level", min_value=0, max_value=3, value=1)
total_working_years = st.number_input("Total Working Years", min_value=0, value=10)
training_times_last_year = st.number_input("Training Times Last Year", min_value=0, value=3)
work_life_balance = st.number_input("Work Life Balance (1-4)", min_value=1, max_value=4, value=3)
years_at_company = st.number_input("Years at Company", min_value=0, value=5)
years_in_current_role = st.number_input("Years in Current Role", min_value=0, value=3)
years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, value=3)

# --------------------------
# Encode categorical values
# --------------------------
input_data = np.array([[
    age,
    business_travel_map[business_travel],
    daily_rate,
    department_map[department],
    distance_from_home,
    education,
    education_field_map[education_field],
    environment_satisfaction,
    gender_map[gender],
    hourly_rate,
    job_involvement,
    job_level,
    job_role_map[job_role],
    job_satisfaction,
    marital_status_map[marital_status],
    monthly_income,
    monthly_rate,
    num_companies_worked,
    overtime_map[overtime],
    percent_salary_hike,
    performance_rating,
    relationship_satisfaction,
    stock_option_level,
    total_working_years,
    training_times_last_year,
    work_life_balance,
    years_at_company,
    years_in_current_role,
    years_since_last_promotion,
    years_with_curr_manager
]])

# --------------------------
# Scale numerical features
# --------------------------
input_data_scaled = scaler.transform(input_data)

# --------------------------
# Predict
# --------------------------
if st.button("Predict Attrition"):
    prediction = model.predict(input_data_scaled)[0]
    st.balloons()
    if prediction == 1:
        st.error("⚠️ The model predicts this employee is likely to LEAVE.")
    else:
        st.success("✅ The model predicts this employee is likely to STAY.")
