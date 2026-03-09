# Credit Risk Default Prediction

Work in progress — building a machine learning model to predict loan default risk.

## Status
- [x] Repo created
- [x] Data acquired
- [x] Exploratory analysis
- [x] Model built
- [ ] Results & write-up

## Project Plan

### Business Problem
Lenders lose significant money when borrowers default on loans — losing the principal, 
interest, and debt recovery costs. This project builds a model to help lenders assess 
the likelihood of default at the point of application, supporting better decisions on 
whether to approve a loan and on what terms.

### What Does Success Look Like?
Missing a genuine defaulter (false negative) is more costly than wrongly flagging a 
good borrower (false positive). A rejected good borrower can appeal or apply elsewhere 
— a defaulted loan cannot be undone. Our model will therefore prioritise catching 
defaulters, and we will evaluate it using metrics that reflect this, not just accuracy.

### What Data Do We Need?
- Annual income and employment history
- Loan amount and interest rate
- Credit score and history of repaying debt
- Current debt obligations (debt-to-income ratio)
- Loan purpose

### Limitations
- Personal circumstances (dependants, cost of living, job stability) are not captured 
  in typical loan datasets
- Loans in the same region are not truly independent — a local economic shock or 
  disaster affects many borrowers simultaneously
- Historical bias: if past lending decisions were discriminatory, the model will learn 
  and perpetuate those patterns

### Plan of Attack
1. Load and inspect the data
2. Clean the data (handle missing values, fix data types)
3. Explore the data — visualisations, patterns, relationships
4. Prepare features for modelling (encode categories, scale numbers)
5. Train two models — Logistic Regression and Random Forest
6. Evaluate and compare the models
7. Write up findings in the README

## Key EDA Findings

- **Loan grade** is the strongest visual predictor of default — 
  default rate rises from 6% (grade A) to 50% (grade G)
- **Interest rate** shows clear separation between defaulters 
  (mean 15.7%) and fully paid loans (mean 12.6%)
- **FICO score** differences are smaller than expected — only a 
  10 point mean difference, with heavy overlap between groups
- **DTI ratio** shows modest separation (17.8 vs 20.2 mean) 
  but distributions overlap heavily
- **Loan purpose** matters — small business loans default at 
  nearly 30%, almost double the overall rate of 20%
- **Delinquencies (2yr)** show weak positive signal but noisy 
  at higher values due to small sample sizes — feature retained 
  for modelling

  ## Data Quality & Limitations

- **Negative DTI values:** 2 records contained negative DTI ratios, 
  likely data entry errors. Retained as impact on model training 
  is negligible (<0.001% of data)
- **emp_length missing values:** 5.8% of records had no employment 
  length recorded — imputed with median rather than mean due to 
  right-skewed distribution
- **Synthetic personal data:** Features capturing personal 
  circumstances (dependants, cost of living, job stability) are 
  absent from the dataset — limiting the model's ability to capture 
  the full picture of borrower risk
- **Temporal independence:** Loans from the same region and time 
  period are not truly independent — a local economic shock affects 
  many borrowers simultaneously
- **Historical bias:** If past lending decisions reflected 
  discriminatory practices, the model may learn and perpetuate 
  those patterns