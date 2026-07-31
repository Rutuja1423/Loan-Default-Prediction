import os
import sqlite3
import pandas as pd

def build_database(cleaned_csv_path, db_path):
    print("=" * 60)
    print("STEP 3A: BUILDING NORMALIZED SQL DATABASE (ETL PIPELINE)")
    print("=" * 60)
    
    df = pd.read_csv(cleaned_csv_path)
    print(f"Loaded {len(df)} records for SQLite ingestion.")
    
    # Remove existing db if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 2. Create Schema
    cursor.execute("""
    CREATE TABLE customers (
        CustomerID TEXT PRIMARY KEY,
        Age INTEGER,
        Income REAL,
        Education TEXT,
        EmploymentType TEXT,
        MaritalStatus TEXT,
        HasMortgage TEXT,
        HasDependents TEXT
    );
    """)
    
    cursor.execute("""
    CREATE TABLE loan_applications (
        LoanID TEXT PRIMARY KEY,
        CustomerID TEXT,
        LoanAmount REAL,
        LoanPurpose TEXT,
        LoanTerm INTEGER,
        InterestRate REAL,
        HasCoSigner TEXT,
        "Default" INTEGER,
        FOREIGN KEY (CustomerID) REFERENCES customers(CustomerID)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE loan_details (
        LoanID TEXT PRIMARY KEY,
        CreditScore INTEGER,
        MonthsEmployed INTEGER,
        NumCreditLines INTEGER,
        DTIRatio REAL,
        FOREIGN KEY (LoanID) REFERENCES loan_applications(LoanID)
    );
    """)
    conn.commit()
    print("Created database schema with tables: customers, loan_applications, loan_details.")
    
    # 3. Populate Tables
    df['CustomerID'] = 'CUST_' + df['LoanID']
    
    # Customers table
    customers_df = df[['CustomerID', 'Age', 'Income', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents']].drop_duplicates()
    customers_df.to_sql('customers', conn, if_exists='append', index=False)
    
    # Loan Applications table
    loans_df = df[['LoanID', 'CustomerID', 'LoanAmount', 'LoanPurpose', 'LoanTerm', 'InterestRate', 'HasCoSigner', 'Default']]
    loans_df.to_sql('loan_applications', conn, if_exists='append', index=False)
    
    # Loan Details table
    details_df = df[['LoanID', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'DTIRatio']]
    details_df.to_sql('loan_details', conn, if_exists='append', index=False)
    
    conn.commit()
    
    # Verification
    for table in ['customers', 'loan_applications', 'loan_details']:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"Table '{table}': {count:,} rows inserted.")
        
    conn.close()
    print(f"Database build complete! File: {db_path}")

if __name__ == "__main__":
    csv_path = os.path.join("data", "cleaned", "loan_data_cleaned.csv")
    database_path = os.path.join("sql", "loan_database.db")
    build_database(csv_path, database_path)
