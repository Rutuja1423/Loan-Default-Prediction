import os
import sqlite3
import pandas as pd

def run_sql_analysis(db_path, dashboard_dir):
    print("=" * 60)
    print("STEP 3B: SQL DATA ANALYSIS & DASHBOARD DATA EXPORT")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}. Please run create_database.py first.")
        
    os.makedirs(dashboard_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Query 1: Overall Summary KPIs
    q1 = """
    SELECT 
        COUNT(*) AS Total_Applications,
        SUM(CASE WHEN "Default" = 0 THEN 1 ELSE 0 END) AS Non_Default_Count,
        SUM(CASE WHEN "Default" = 1 THEN 1 ELSE 0 END) AS Default_Count,
        ROUND(AVG("Default") * 100.0, 2) AS Default_Rate_Pct,
        ROUND(AVG(LoanAmount), 2) AS Avg_Loan_Amount,
        ROUND(AVG(InterestRate), 2) AS Avg_Interest_Rate
    FROM loan_applications;
    """
    kpi_df = pd.read_sql_query(q1, conn)
    print("\n--- QUERY 1: OVERALL LOAN KPI SUMMARY ---")
    print(kpi_df.to_string(index=False))
    kpi_df.to_csv(os.path.join(dashboard_dir, "sql_kpi_summary.csv"), index=False)
    
    # Query 2: Education Analysis
    q2 = """
    SELECT 
        c.Education,
        COUNT(l.LoanID) AS Total_Loans,
        SUM(l."Default") AS Default_Count,
        ROUND(AVG(l."Default") * 100.0, 2) AS Default_Rate_Pct,
        ROUND(AVG(c.Income), 2) AS Avg_Income,
        ROUND(AVG(l.LoanAmount), 2) AS Avg_Loan_Amount
    FROM loan_applications l
    JOIN customers c ON l.CustomerID = c.CustomerID
    GROUP BY c.Education
    ORDER BY Default_Rate_Pct DESC;
    """
    edu_df = pd.read_sql_query(q2, conn)
    print("\n--- QUERY 2: DEFAULT BY EDUCATION LEVEL ---")
    print(edu_df.to_string(index=False))
    edu_df.to_csv(os.path.join(dashboard_dir, "sql_default_by_education.csv"), index=False)
    
    # Query 3: Employment Analysis
    q3 = """
    SELECT 
        c.EmploymentType,
        COUNT(l.LoanID) AS Total_Loans,
        SUM(l."Default") AS Default_Count,
        ROUND(AVG(l."Default") * 100.0, 2) AS Default_Rate_Pct,
        ROUND(AVG(c.Income), 2) AS Avg_Income,
        ROUND(AVG(d.MonthsEmployed), 1) AS Avg_Months_Employed
    FROM loan_applications l
    JOIN customers c ON l.CustomerID = c.CustomerID
    JOIN loan_details d ON l.LoanID = d.LoanID
    GROUP BY c.EmploymentType
    ORDER BY Default_Rate_Pct DESC;
    """
    emp_df = pd.read_sql_query(q3, conn)
    print("\n--- QUERY 3: DEFAULT BY EMPLOYMENT TYPE ---")
    print(emp_df.to_string(index=False))
    emp_df.to_csv(os.path.join(dashboard_dir, "sql_default_by_employment.csv"), index=False)
    
    # Query 4: Loan Purpose Analysis
    q4 = """
    SELECT 
        l.LoanPurpose,
        COUNT(l.LoanID) AS Total_Loans,
        SUM(l."Default") AS Default_Count,
        ROUND(AVG(l."Default") * 100.0, 2) AS Default_Rate_Pct,
        ROUND(AVG(l.LoanAmount), 2) AS Avg_Loan_Amount,
        ROUND(AVG(l.InterestRate), 2) AS Avg_Interest_Rate
    FROM loan_applications l
    GROUP BY l.LoanPurpose
    ORDER BY Default_Rate_Pct DESC;
    """
    purpose_df = pd.read_sql_query(q4, conn)
    print("\n--- QUERY 4: DEFAULT BY LOAN PURPOSE ---")
    print(purpose_df.to_string(index=False))
    purpose_df.to_csv(os.path.join(dashboard_dir, "sql_default_by_purpose.csv"), index=False)

    # Query 5: Defaulter vs Non-Defaulter Financial Profiles
    q5 = """
    SELECT 
        CASE WHEN l."Default" = 1 THEN 'Defaulter' ELSE 'Non-Defaulter' END AS Status,
        COUNT(l.LoanID) AS Applicant_Count,
        ROUND(AVG(c.Income), 2) AS Avg_Income,
        ROUND(AVG(l.LoanAmount), 2) AS Avg_Loan_Amount,
        ROUND(AVG(d.CreditScore), 1) AS Avg_Credit_Score,
        ROUND(AVG(d.DTIRatio), 3) AS Avg_DTI_Ratio,
        ROUND(AVG(l.InterestRate), 2) AS Avg_Interest_Rate
    FROM loan_applications l
    JOIN customers c ON l.CustomerID = c.CustomerID
    JOIN loan_details d ON l.LoanID = d.LoanID
    GROUP BY l."Default";
    """
    profile_df = pd.read_sql_query(q5, conn)
    print("\n--- QUERY 5: FINANCIAL PROFILES COMPARISON ---")
    print(profile_df.to_string(index=False))
    profile_df.to_csv(os.path.join(dashboard_dir, "sql_financial_profiles.csv"), index=False)

    conn.close()
    print("\nSQL Analysis complete! Results exported to CSVs in:", dashboard_dir)

if __name__ == "__main__":
    db_file = os.path.join("sql", "loan_database.db")
    dash_dir = "dashboard"
    run_sql_analysis(db_file, dash_dir)
