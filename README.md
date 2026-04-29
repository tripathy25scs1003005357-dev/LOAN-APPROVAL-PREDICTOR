# Loan Approval Predictor

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details such as income, credit history, loan amount, education, employment status, and property area.

## Project Overview

Loan approval is an important process in the banking and finance sector. This project helps automate loan eligibility prediction using data analysis and machine learning techniques.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Loan approval prediction using ML model
- User-friendly prediction system
- Suitable for beginner data science portfolio

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Dataset Columns

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status

## Machine Learning Workflow

1. Import dataset
2. Handle missing values
3. Encode categorical columns
4. Split data into training and testing sets
5. Train ML model
6. Evaluate accuracy
7. Predict loan approval result

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
streamlit run app.py
