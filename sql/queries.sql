-- ============================================================
-- SQL ANALYTICAL QUERIES FOR LOAN DEFAULT RISK ANALYSIS
-- ============================================================

-- Query 1: Total Applications, Total Approved (Non-Default), Total Defaulted, Default Rate
SELECT 
    COUNT(*) AS Total_Applications,
    SUM(CASE WHEN "Default" = 0 THEN 1 ELSE 0 END) AS Non_Default_Count,
    SUM(CASE WHEN "Default" = 1 THEN 1 ELSE 0 END) AS Default_Count,
    ROUND(AVG("Default") * 100.0, 2) AS Default_Rate_Pct
FROM loan_applications;

-- Query 2: Default Rate & Average Metrics by Education Level
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

-- Query 3: Default Rate & Metrics by Employment Type
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

-- Query 4: Default Rate & Average Loan Amount by Loan Purpose
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

-- Query 5: Comparison of Defaulters vs Non-Defaulters Key Financial Indicators
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

-- Query 6: Default Risk by Credit Score Bands
SELECT 
    CASE 
        WHEN d.CreditScore < 580 THEN '1. Poor (<580)'
        WHEN d.CreditScore BETWEEN 580 AND 669 THEN '2. Fair (580-669)'
        WHEN d.CreditScore BETWEEN 670 AND 739 THEN '3. Good (670-739)'
        WHEN d.CreditScore BETWEEN 740 AND 799 THEN '4. Very Good (740-799)'
        ELSE '5. Excellent (800+)'
    END AS Credit_Score_Band,
    COUNT(l.LoanID) AS Total_Loans,
    SUM(l."Default") AS Default_Count,
    ROUND(AVG(l."Default") * 100.0, 2) AS Default_Rate_Pct
FROM loan_applications l
JOIN loan_details d ON l.LoanID = d.LoanID
GROUP BY Credit_Score_Band
ORDER BY Credit_Score_Band;

-- Query 7: High-Risk Applicants Segment (High DTI > 0.6 AND Credit Score < 600)
SELECT 
    COUNT(l.LoanID) AS High_Risk_Count,
    SUM(l."Default") AS High_Risk_Defaults,
    ROUND(AVG(l."Default") * 100.0, 2) AS High_Risk_Default_Rate_Pct
FROM loan_applications l
JOIN loan_details d ON l.LoanID = d.LoanID
WHERE d.DTIRatio > 0.6 AND d.CreditScore < 600;

-- Query 8: Default Rate by Co-Signer Presence
SELECT 
    l.HasCoSigner,
    COUNT(l.LoanID) AS Total_Loans,
    SUM(l."Default") AS Default_Count,
    ROUND(AVG(l."Default") * 100.0, 2) AS Default_Rate_Pct
FROM loan_applications l
GROUP BY l.HasCoSigner;
