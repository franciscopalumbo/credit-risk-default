import pandas as pd

df = pd.read_csv('data/accepted_2007_to_2018Q4.csv.gz', 
                 compression='gzip', 
                 low_memory=False)

# Keep only loans with a known final outcome
keep_status = [
    'Fully Paid',
    'Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off',
    'Default'
]

df_clean = df[df['loan_status'].isin(keep_status)].copy()

# Create target variable
df_clean['default'] = df_clean['loan_status'].isin([
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
]).astype(int)

# Keep only the features to use in model, and check for missing values
FEATURES = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc', 'purpose',
    'dti', 'delinq_2yrs', 'fico_range_low', 'open_acc',
    'pub_rec', 'revol_util'
]

df_model = df_clean[FEATURES + ['default']].copy()

# Drop rows where missing values are negligible
df_model.dropna(subset=[
    'revol_util', 'dti', 'annual_inc', 
    'delinq_2yrs', 'open_acc', 'pub_rec'
], inplace=True)

# Convert emp_length from string to numeric
# '< 1 year' → 0, '1 year' → 1, '10+ years' → 10
df_model['emp_length'] = df_model['emp_length'].str.replace("< 1 year", "0")
df_model['emp_length'] = df_model['emp_length'].str.extract(r'(\d+)')
df_model['emp_length'] = df_model['emp_length'].astype(float)

# emp_length: 5.8% missing — impute with median rather than mean
# as distribution is right-skewed by long-tenured borrowers
df_model['emp_length'].fillna(df_model['emp_length'].median(), inplace=True)

# Verify no missing values remain
print(df_model.isnull().sum())
print(f"Rows remaining: {len(df_model):,}")