# Data Dictionary

| Column | Description |
|---|---|
| applicant_id | Unique applicant identifier |
| age | Applicant age in years |
| gender | Gender label |
| state | Indian state code |
| city_tier | Urban tier label: Tier 1/2/3 |
| rural_flag | 1 if applicant is from a rural area, else 0 |
| pincode | Applicant pincode |
| annual_income | Annual income in INR |
| employment_type | Salaried, Self-Employed, or Contract |
| employer_tier | Employer quality tier A/B/C |
| education | Education level |
| marital_status | Single, Married, or Separated |
| num_dependents | Number of dependents |
| existing_loans | Count of existing loans |
| credit_bureau_score | Credit bureau score |
| bank_relationship_years | Years of relationship with the bank |
| digital_transactions | Monthly digital transaction value proxy |
| mobile_os | Mobile operating system |
| loan_amount | Requested loan amount |
| loan_purpose | Loan purpose category |
| credit_score | Legacy credit score field |
| model_score | Fairness/ML score variant |
| credit_ai_score | AI credit model score |
| credit_ai_decision | AI-driven default/approval decision flag |
| actual_default_6m | Ground-truth 6-month default label |
| six_month_default | Backward-compatible default label |
| credit_decision | Final credit decision (0/1) |
