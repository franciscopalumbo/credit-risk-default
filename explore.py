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
print(f"Rows before: {len(df):,}")
print(f"Rows after:  {len(df_clean):,}")

# Now create our target variable
df_clean['default'] = df_clean['loan_status'].isin([
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
]).astype(int)

print(df_clean['default'].value_counts())
print(f"Default rate: {df_clean['default'].mean():.1%}")