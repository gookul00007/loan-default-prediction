import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("C:/Data science and AI/Projects/Loan defaulters Prediction/model/model.pkl", "rb"))
preprocessor = pickle.load(open("C:/Data science and AI/Projects/Loan defaulters Prediction/model/preprocessor.pkl", "rb"))


st.set_page_config(page_title="Loan Default Predictor", layout="centered")

st.title("Loan Default Prediction System")
st.write("Enter applicant details to predict default risk")

st.subheader("Applicant Information")

age = st.number_input("Age", 18, 70)
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
months_employed = st.number_input("Months Employed")
num_credit_lines = st.number_input("Number of Credit Lines")
interest_rate = st.number_input("Interest Rate")
loan_term = st.number_input("Loan Term")
dti = st.number_input("DTI Ratio")

education = st.selectbox(
    "Education",
    ["High School", "Bachelor's", "Master's", "PhD"]
)
employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed","Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married","Divorced"])
purpose = st.selectbox(
    "Loan Purpose",
    ["Other", "Auto", "Business", "Home", "Education"]
)

has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
    'Age': float(age),
    'Income': float(income),
    'LoanAmount': float(loan_amount),
    'CreditScore': float(credit_score),
    'MonthsEmployed': float(months_employed),
    'NumCreditLines': float(num_credit_lines),
    'InterestRate': float(interest_rate),
    'LoanTerm': float(loan_term),
    'DTIRatio': float(dti),
    'Education': str(education),
    'EmploymentType': str(employment),
    'MaritalStatus': str(marital),
    'LoanPurpose': str(purpose),
    'HasMortgage': has_mortgage,
    'HasDependents': has_dependents,
    'HasCoSigner': has_cosigner
}])

    processed = preprocessor.transform(input_data)

    prediction = model.predict(processed)
    probability = model.predict_proba(processed)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"High Risk of Default (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk (Probability: {probability:.2f})")