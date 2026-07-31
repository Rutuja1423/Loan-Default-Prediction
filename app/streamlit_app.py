import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-approved {
        background-color: #DCFCE7;
        color: #166534;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
    }
    .status-default {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts safely
@st.cache_resource
def load_model_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    feature_names_path = os.path.join(models_dir, "feature_names.pkl")
    model_name_path = os.path.join(models_dir, "best_model_name.txt")
    
    if not (os.path.exists(model_path) and os.path.exists(feature_names_path)):
        return None, None, None, "Not Trained Yet"
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    feature_names = joblib.load(feature_names_path)
    
    model_name = "Best Model"
    if os.path.exists(model_name_path):
        with open(model_name_path, "r") as f:
            model_name = f.read().strip()
            
    return model, scaler, feature_names, model_name

def calculate_derived_features(input_dict):
    # Calculate derived financial ratios matching feature engineering
    income = input_dict['Income']
    loan_amt = input_dict['LoanAmount']
    interest_rate = input_dict['InterestRate']
    loan_term = input_dict['LoanTerm']
    age = input_dict['Age']
    months_emp = input_dict['MonthsEmployed']
    credit_score = input_dict['CreditScore']
    dti_ratio = input_dict['DTIRatio']
    
    loan_to_income = loan_amt / income if income > 0 else 0
    monthly_income = income / 12.0
    monthly_r = (interest_rate / 100.0) / 12.0
    
    if monthly_r > 0:
        pow_factor = (1 + monthly_r) ** loan_term
        emi = loan_amt * monthly_r * pow_factor / (pow_factor - 1)
    else:
        emi = loan_amt / loan_term
        
    emi_income_ratio = emi / monthly_income if monthly_income > 0 else 0
    adult_months = max((age - 18) * 12, 1)
    emp_ratio = months_emp / adult_months
    
    if credit_score < 580:
        credit_band = 0
    elif credit_score < 670:
        credit_band = 1
    elif credit_score < 740:
        credit_band = 2
    elif credit_score < 800:
        credit_band = 3
    else:
        credit_band = 4
        
    composite_risk = (dti_ratio * interest_rate) / (credit_score / 100.0)
    high_risk_flag = 1 if (dti_ratio > 0.6 and credit_score < 600) else 0
    total_interest = (emi * loan_term) - loan_amt
    
    derived = {
        'LoanToIncomeRatio': loan_to_income,
        'MonthlyIncome': monthly_income,
        'EMI': emi,
        'EMIToIncomeRatio': emi_income_ratio,
        'EmploymentRatio': emp_ratio,
        'CreditScoreBand': credit_band,
        'CompositeRiskIndex': composite_risk,
        'HighRiskFlag': high_risk_flag,
        'TotalInterestCost': total_interest
    }
    return derived

def preprocess_single_input(input_dict, feature_names):
    derived = calculate_derived_features(input_dict)
    full_dict = {**input_dict, **derived}
    
    row = {}
    row['Age'] = full_dict['Age']
    row['Income'] = full_dict['Income']
    row['LoanAmount'] = full_dict['LoanAmount']
    row['CreditScore'] = full_dict['CreditScore']
    row['MonthsEmployed'] = full_dict['MonthsEmployed']
    row['NumCreditLines'] = full_dict['NumCreditLines']
    row['InterestRate'] = full_dict['InterestRate']
    row['LoanTerm'] = full_dict['LoanTerm']
    row['DTIRatio'] = full_dict['DTIRatio']
    
    # Binary maps
    row['HasMortgage'] = 1 if full_dict['HasMortgage'] == 'Yes' else 0
    row['HasDependents'] = 1 if full_dict['HasDependents'] == 'Yes' else 0
    row['HasCoSigner'] = 1 if full_dict['HasCoSigner'] == 'Yes' else 0
    
    # Education
    edu_map = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    row['Education'] = edu_map.get(full_dict['Education'], 1)
    
    # One-hot features
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    for emp in employment_types:
        row[f'EmploymentType_{emp}'] = 1 if full_dict['EmploymentType'] == emp else 0
        
    marital_statuses = ['Divorced', 'Married', 'Single']
    for ms in marital_statuses:
        row[f'MaritalStatus_{ms}'] = 1 if full_dict['MaritalStatus'] == ms else 0
        
    loan_purposes = ['Auto', 'Business', 'Education', 'Home', 'Other']
    for lp in loan_purposes:
        row[f'LoanPurpose_{lp}'] = 1 if full_dict['LoanPurpose'] == lp else 0
        
    # Add derived
    for k, v in derived.items():
        row[k] = v
        
    df_row = pd.DataFrame([row])
    # Reindex to exact feature names list
    df_row = df_row.reindex(columns=feature_names, fill_value=0)
    return df_row, derived

def main():
    st.markdown('<div class="main-header">🏦 AI Loan Default Prediction & Credit Risk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Decision System & Risk Scorecard for Financial Institutions</div>', unsafe_allow_html=True)
    
    model, scaler, feature_names, model_name = load_model_artifacts()
    
    if model is None:
        st.warning("⚠️ ML Model artifacts not found. Please run the model training pipeline first (`python main.py`).")
        st.info("Showing UI layout for demonstration.")
        return

    st.sidebar.markdown(f"**Loaded Model:** `{model_name}`")
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Borrower Input Form")
    
    # Input controls in sidebar
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
    loan_amount = st.sidebar.number_input("Loan Amount Requested ($)", min_value=1000, max_value=500000, value=25000, step=2500)
    credit_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=680)
    months_employed = st.sidebar.number_input("Months Employed", min_value=0, max_value=600, value=48)
    num_credit_lines = st.sidebar.slider("Number of Credit Lines", min_value=1, max_value=20, value=4)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, max_value=35.0, value=10.5, step=0.5)
    loan_term = st.sidebar.selectbox("Loan Term (Months)", options=[12, 24, 36, 48, 60], index=2)
    dti_ratio = st.sidebar.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    
    st.sidebar.markdown("---")
    education = st.sidebar.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD"], index=1)
    employment_type = st.sidebar.selectbox("Employment Type", options=["Full-time", "Part-time", "Self-employed", "Unemployed"], index=0)
    marital_status = st.sidebar.selectbox("Marital Status", options=["Single", "Married", "Divorced"], index=1)
    loan_purpose = st.sidebar.selectbox("Loan Purpose", options=["Auto", "Business", "Education", "Home", "Other"], index=0)
    has_mortgage = st.sidebar.radio("Has Existing Mortgage?", options=["No", "Yes"], index=0)
    has_dependents = st.sidebar.radio("Has Dependents?", options=["No", "Yes"], index=0)
    has_cosigner = st.sidebar.radio("Has Co-Signer?", options=["Yes", "No"], index=0)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Application Assessment", "📊 Model Insights & Features", "📁 Batch Prediction"])
    
    with tab1:
        st.subheader("Real-Time Application Risk Evaluation")
        
        input_data = {
            'Age': age, 'Income': income, 'LoanAmount': loan_amount,
            'CreditScore': credit_score, 'MonthsEmployed': months_employed,
            'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
            'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
            'Education': education, 'EmploymentType': employment_type,
            'MaritalStatus': marital_status, 'LoanPurpose': loan_purpose,
            'HasMortgage': has_mortgage, 'HasDependents': has_dependents,
            'HasCoSigner': has_cosigner
        }
        
        X_single, derived = preprocess_single_input(input_data, feature_names)
        
        if st.sidebar.button("🚀 Evaluate Loan Application", use_container_width=True):
            if model_name == 'Logistic Regression' and scaler is not None:
                X_input = scaler.transform(X_single)
            else:
                X_input = X_single
                
            prob_default = float(model.predict_proba(X_input)[0, 1])
            prob_approve = (1.0 - prob_default) * 100.0
            prob_default_pct = prob_default * 100.0
            
            # Risk tier determination
            if prob_default < 0.30:
                risk_tier = "Low Risk"
                status_class = "status-approved"
                status_msg = "✅ LOAN APPROVED"
                recommendation = "Low default probability. Fast-track approval recommended."
            elif prob_default < 0.60:
                risk_tier = "Medium Risk"
                status_class = "status-approved"
                status_msg = "⚠️ CONDITIONAL APPROVAL / MANUAL REVIEW"
                recommendation = "Moderate risk. Additional proof of income or higher down payment advised."
            else:
                risk_tier = "High Risk"
                status_class = "status-default"
                status_msg = "❌ HIGH DEFAULT RISK (REJECT / MANUAL REVIEW)"
                recommendation = "Elevated risk of default. Requires underwriting committee manual review."

            col1, col2, col3 = st.columns([1.5, 1, 1])
            
            with col1:
                st.markdown(f'<div class="{status_class}">{status_msg}</div>', unsafe_allow_html=True)
                st.markdown(f"**Recommendation:** {recommendation}")
                
            with col2:
                st.metric(label="Default Probability", value=f"{prob_default_pct:.1f}%")
                st.metric(label="Risk Classification Tier", value=risk_tier)
                
            with col3:
                st.metric(label="Approval Confidence", value=f"{prob_approve:.1f}%")
                st.metric(label="Monthly EMI", value=f"${derived['EMI']:,.2f}")

            st.markdown("---")
            st.subheader("💡 Financial & Risk Metrics Breakdown")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Loan-to-Income Ratio", f"{derived['LoanToIncomeRatio']:.2f}")
            mcol2.metric("EMI-to-Monthly-Income", f"{derived['EMIToIncomeRatio']:.1%}")
            mcol3.metric("Composite Risk Index", f"{derived['CompositeRiskIndex']:.2f}")
            mcol4.metric("Total Interest Payable", f"${derived['TotalInterestCost']:,.2f}")

    with tab2:
        st.subheader("Model Performance & Feature Importance")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        comp_csv = os.path.join(base_dir, "dashboard", "model_comparison.csv")
        fi_csv = os.path.join(base_dir, "dashboard", "feature_importance.csv")
        
        if os.path.exists(comp_csv):
            st.markdown("### 🏆 Model Comparison Matrix")
            m_df = pd.read_csv(comp_csv)
            st.dataframe(m_df, use_container_width=True)
            
        if os.path.exists(fi_csv):
            st.markdown("### 🔑 Key Drivers Influencing Default Predictions")
            fi_df = pd.read_csv(fi_csv).head(15)
            st.bar_chart(data=fi_df.set_index('Feature')['Importance'])

    with tab3:
        st.subheader("📁 Batch Loan Application Processing")
        st.write("Upload a CSV file containing multiple loan applications to run batch predictions.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded dataset contains {len(batch_df):,} rows.")
            st.dataframe(batch_df.head())

if __name__ == "__main__":
    main()
