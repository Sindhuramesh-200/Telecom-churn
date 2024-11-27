import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model, scaler, and encoder
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('s.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ohe.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Streamlit App Config
st.set_page_config(page_title="Telecom Churn Prediction App", page_icon="☎️", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Telecom Churn Prediction App</h1>", unsafe_allow_html=True)

st.write('---')

# Customer Input Section
name = st.text_input('Enter your name:')
st.subheader(f'Hi {name}, fill in your details below to predict churn status.')

st.markdown(f'''The Telecom Churn Prediction App analyses customer behaviour to predict churn. Users enter appropriate selections to get results. The findings are obtained using a trained model, which provides forecasts. These forecasts are the result of training the model with qualities and characteristics from both churners and non-churners.''')

# Collect user inputs
gender = st.radio('Select Gender:', ('Male', 'Female'))
senior_citizen = st.radio('Are you a Senior Citizen?', ('No', 'Yes'))
partner = st.radio('Do you have a partner?', ('Yes', 'No'))
dependents = st.radio('Do you have any dependents?', ('Yes', 'No'))
phone_service = st.radio('Do you have phone service?', ('Yes', 'No'))
multiple_lines = st.radio('Do you use multiple lines?', ('Yes', 'No'))
internet_service = st.selectbox('Select Internet Service:', ['DSL', 'Fiber optic', 'No'])
online_security = st.radio('Do you use online security?', ('Yes', 'No'))
online_backup = st.radio('Do you use online backup?', ('Yes', 'No'))
device_protection = st.radio('Do you use device protection?', ('Yes', 'No'))
tech_support = st.radio('Do you use tech support?', ('Yes', 'No'))
streaming_tv = st.radio('Do you stream TV?', ('Yes', 'No'))
streaming_movies = st.radio('Do you stream movies?', ('Yes', 'No'))
contract = st.selectbox("Select Contract Type:", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.radio('Do you use paperless billing?', ('Yes', 'No'))
payment_method = st.selectbox('Select Payment Method:', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
tenure = st.slider('How many months have you been with the service?', 0, 72, 24)
monthly_charges = st.slider('Monthly Charges (USD):', 0, 150, 70)
total_charges = st.slider('Total Charges (USD):', 0, 9000, 2500)

# Create input dataframe
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Columns for numerical and categorical features
numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod']

# Preprocess input
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
input_data_encoded = encoder.transform(input_data[categorical_cols])
input_ohe_df = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Final input dataframe with numerical and one-hot encoded categorical columns
input_final = pd.concat([input_data[numerical_cols].reset_index(drop=True), input_ohe_df], axis=1)

# Ensure correct column order
input_final = input_final.reindex(columns=feature_names, fill_value=0)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_final)

    if prediction[0] == 1:
        st.error("The customer is likely to churn!")
    else:
        st.success("The customer is likely to stay!")

# Hide Streamlit's default menu and footer
hide_menu_style = '''
       <style>
       #MainMenu {visibility: hidden;}
       footer {visibility: hidden;}
       </style>
        '''
st.markdown(hide_menu_style, unsafe_allow_html=True)
