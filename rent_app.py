# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and sample input
model = joblib.load('Model/ucc_hostel_rent_predictor.pkl')
sample_input = joblib.load('Model/hostel_sample_input.pkl')

# Streamlit app title
#st.title("UCC Hostel Rent Predictor APP")
#st.write("Enter the hostel details to predict the annual rent.")

# Configure the Streamlit app
st.set_page_config(page_title="UCC Hostel Rent Predictor APP", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["🏠Home", "Predict Hostel Rent"])

# ------------------ HOME PAGE ------------------ #
if page == "🏠Home":
    st.title("🏠 Hostel Rent Prediction App ")

    st.markdown("""
Welcome to the **Hostel Rent Predictor** designed for students and stakeholders of the **University of Cape Coast (UCC)**.

This app allows:
- 🎓 **Students** to estimate hostel rent based on room features, amenities, and location.
- 🧠 **Researchers** to explore data patterns in student accommodation pricing.
- 🏢 **Administrators and hostel owners** to understand factors that influence rent.

---

### 👨‍💻 Authors

---

#### **Mohammed Kamalidin**  
📧 [mkamalidin9@gmail.com](mailto:mkamalidin9@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/mohammed-kamalidin/)  
🎓 University of Cape Coast  
📘 MSc Data Management and Analysis  
🏢 Assistant Technical Officer @ National Identification Authority

---

#### **Prince Acquah Rockson**  
📧 [parockson@gmail.com](mailto:parockson@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/prince-acquah-rockson/)  
🎓 Kwame Nkrumah University of Science and Technology  
📘 MSc Health Informatics  
💼 Data Analyst @ Halges Fintech  
💻 Software Engineer @ Proxy Solutions
    """)

# ------------------ PREDICT PAGE ------------------ #
elif page == "Predict Hostel Rent":
    st.title("🏘️ Predict Hostel Rent")
    # Create input fields based on feature types
    st.markdown("Please provide the details below:")
    #st.header("Hostel Details")
    
    # Numerical features
    st.subheader("Numerical Features")
    deposit = st.number_input("Required Deposit (GHS)", min_value=0.0, value=2500.0)
    rent_increase = st.number_input("Recent Rent Increase (GHS)", min_value=0.0, value=500.0)
    avg_area_rent = st.number_input("Average Rent Nearby (GHS)", min_value=0.0, value=4000.0)
    commute_time = st.number_input("Commute Time (minutes)", min_value=0.0, value=15.0)
    room_size = st.number_input("Room Size (sqm)", min_value=0.0, value=20.0)
    furnishing_score = st.slider("Furnishing Score (0-3)", 0, 3, 2)
    total_amenities = st.slider("Total Amenities (0-11)", 0, 11, 5)
    rent_to_deposit_ratio = st.number_input("Rent to Deposit Ratio", min_value=0.0, value=1.5)
    distance_weighted_rent = st.number_input("Distance-Weighted Rent", min_value=0.0, value=200.0)
    log_avg_area_rent = np.log1p(avg_area_rent)
    log_deposit = np.log1p(deposit)
    
    # Categorical features
    st.subheader("Categorical Features")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age_group = st.selectbox("Age Group", ['18-25', '25-30', '26-30', '35-40'])
    study_level = st.selectbox("Study Level", ['First year', 'Second year', 'Third year', 'Fourth year', 'Postgraduate'])
    campus_location = st.selectbox("Campus Location", ['Science', 'New Site'])
    room_type = st.selectbox("Room Type", ['Private room(1 in a room)', '2 in a room','3 in a room','4 in a room', '5+ in a room'])
    faculty = st.selectbox("Faculty", ['Faculty of Social Sciences', 'School of Business', 'Faculty of Education'])
    stay_duration = st.selectbox("Stay Duration", ['Less than 6 months', '6 months to 1 year', 'Between a year and 2 years', '3 years or more'])
    room_category = st.selectbox("Room Category", ['Shared washroom- shared kitchen', 'Full self contain', 'Shared washroom- No kitchen'])
    hostel_location = st.selectbox("Hostel Location", ['Apewosika', 'Kwaprow', 'Amamoma', 'Domeabra','Abura'])
    commute_mode = st.selectbox("Commute Mode", ['Walking', 'Public Transport', 'Private Vehicle'])
    
    # Binary features
    st.subheader("Amenities (Yes/No)")
    water_included = st.checkbox("Water Included")
    electricity_included = st.checkbox("Electricity Included")
    waste_disposal_included = st.checkbox("Waste Disposal Included")
    running_water = st.checkbox("Running Water")
    extra_storage = st.checkbox("Extra Storage")
    wifi = st.checkbox("Wi-Fi")
    study_area = st.checkbox("Study Area")
    security = st.checkbox("Security Services")
    generator_backup = st.checkbox("Generator Backup")
    access_control = st.checkbox("Access Control")
    janitorial_services = st.checkbox("Janitorial Services")
    
    # Create input dictionary
    input_data = {
        'gender': gender,
        'age_group': age_group,
        'study_level': study_level,
        'campus_location': campus_location,
        'room_type': room_type,
        'faculty': faculty,
        'stay_duration': stay_duration,
        'room_category': room_category,
        'water_included': 1 if water_included else 0,
        'electricity_included': 1 if electricity_included else 0,
        'waste_disposal_included': 1 if waste_disposal_included else 0,
        'running_water': 1 if running_water else 0,
        'extra_storage': 1 if extra_storage else 0,
        'wifi': 1 if wifi else 0,
        'study_area': 1 if study_area else 0,
        'security': 1 if security else 0,
        'generator_backup': 1 if generator_backup else 0,
        'commute_time': commute_time,
        'commute_mode': commute_mode,
        'room_size': room_size,
        'access_control': 1 if access_control else 0,
        'janitorial_services': 1 if janitorial_services else 0,
        'deposit': deposit,
        'rent_increase': rent_increase,
        'avg_area_rent': avg_area_rent,
        'hostel_location': hostel_location,
        'furnishing_score': furnishing_score,
        'total_amenities': total_amenities,
        'rent_to_deposit_ratio': rent_to_deposit_ratio,
        'distance_weighted_rent': distance_weighted_rent,
        'log_avg_area_rent': log_avg_area_rent,
        'log_deposit': log_deposit
    }
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict rent
    if st.button("Predict Rent"):
        pred_log = model.predict(input_df)
        pred_rent = np.expm1(pred_log)[0]
        st.success(f"Predicted Annual Rent: GHS {pred_rent:.2f}")
    
    # Display sample input for reference
    #if st.checkbox("Show Sample Input"):
    #    st.write("Sample Input Data:")
    #    st.write(sample_input)
