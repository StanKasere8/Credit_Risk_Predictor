import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model we trained in the notebook
# Using joblib lets us load the saved 'brain' file directly
model = joblib.load('credit_model.pkl')

# Set up the page layout and icon so it looks professional in the browser tab
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")

# Main title and description
st.title("💳 Credit Risk Default Predictor")
st.write("Adjust the financial parameters below to see how they impact the risk score.")

# --- User Inputs ---
# We arrange these in two columns to save vertical space on the screen
col1, col2 = st.columns(2)

with col1:
    # Age and Income are basic demographics
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
    
    # Debt ratio is critical - higher is usually worse
    debt_ratio = st.slider("Debt Ratio (Debt / Income)", 0.0, 10.0, 0.5, step=0.01)
    
    # Utilization refers to how much of their credit limit they are using
    utilization = st.slider("Revolving Utilization (0-1)", 0.0, 2.0, 0.1, step=0.01)

with col2:
    # Late payment history is usually the strongest predictor of risk
    times_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=0)
    times_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0)
    times_90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)
    
    # Number of open lines (credit cards, car loans, etc.)
    open_loans = st.number_input("Number of Open Loans", min_value=0, max_value=50, value=5)
    
    # Specific real estate exposure
    real_estate_loans = st.number_input("Real Estate Loans", min_value=0, max_value=20, value=1)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

# --- Prediction Logic ---
if st.button("Predict Credit Risk"):
    
    # Show a spinner so the user knows it's calculating
    with st.spinner('Calculating risk score...'):
        
        # The model expects a pandas DataFrame with specific column names.
        # The order of columns here MUST match exactly what we trained on, 
        # otherwise the model will get confused.
        input_data = pd.DataFrame({
            'RevolvingUtilizationOfUnsecuredLines': [utilization],
            'age': [age],
            'NumberOfTime30-59DaysPastDueNotWorse': [times_30_59],
            'DebtRatio': [debt_ratio],
            'MonthlyIncome': [monthly_income],
            'NumberOfOpenCreditLinesAndLoans': [open_loans],
            'NumberOfTimes90DaysLate': [times_90],
            'NumberRealEstateLoansOrLines': [real_estate_loans],
            'NumberOfTime60-89DaysPastDueNotWorse': [times_60_89],
            'NumberOfDependents': [dependents]
        })

        # Get the prediction (0 or 1) and the probability (confidence level)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # --- Display Results ---
        # prediction[0] is the result. 1 = Default, 0 = Safe.
        if prediction[0] == 1:
            st.error(f"⚠️ High Risk of Default!")
            st.write(f"Probability of Default: {probability[0][1]:.2%}")
        else:
            st.success(f"✅ Low Risk: Loan Approved")
            # We still show the probability even for safe loans
            st.write(f"Probability of Default: {probability[0][1]:.2%}")