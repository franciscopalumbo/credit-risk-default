import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv('data/loans_clean.csv')

# ── Chart 1: Target Distribution ──────────────────────
# Question: How imbalanced is our target variable?

sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(6, 4))

counts = df['default'].value_counts()
ax.bar(['Fully Paid', 'Default'], counts.values, 
       color=['#2ECC71', '#E74C3C'])

ax.set_title('Loan Outcome Distribution')
ax.set_ylabel('Number of Loans')

# Format y-axis to show full numbers instead of scientific notation
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{int(x):,}')
)

# Add the count and percentage on top of each bar
for bar, count in zip(ax.patches, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 10000,
            f'{count:,}\n({count/len(df):.1%})',
            ha='center', va='bottom', fontsize=11)
    
ax.set_ylim(0, counts.max() * 1.2)

plt.tight_layout()
plt.savefig('outputs/figures/01_target_distribution.png')
plt.close()

print(counts)
print(f"Default rate: {df['default'].mean():.1%}")

# ── Chart 2: Default Rate by Loan Grade ───────────────
# Question: Does default rate rise consistently from grade A to G?

grade_stats = df.groupby('grade')['default'].mean().reset_index()
grade_stats.columns = ['grade', 'default_rate']
grade_stats = grade_stats.sort_values('grade')

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='grade', y='default_rate', data=grade_stats,
            palette='Reds', ax=ax)
ax.set_title('Default Rate by Loan Grade')
ax.set_ylabel('Default Rate')
ax.set_xlabel('Loan Grade')

counts = df['grade'].value_counts().sort_index()

# Add the percentage on top of each bar
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.1%}',
            ha='center', va='bottom', fontsize=11)
ax.set_ylim(0, grade_stats['default_rate'].max() * 1.2)

plt.tight_layout()
plt.savefig('outputs/figures/02_default_by_grade.png')
plt.close()

# ── Chart 3: FICO Score Distribution ──────────────────
# Question: Do defaulters tend to have lower FICO scores?

fig, ax = plt.subplots(figsize=(8, 5))

for label, colour, name in [(0, '#2ECC71', 'Fully Paid'), 
                             (1, '#E74C3C', 'Default')]:
    subset = df[df['default'] == label]['fico_range_low']
    ax.hist(subset, bins=40, alpha=0.6, label=name, color=colour, density=True)
    ax.axvline(subset.mean(), color=colour, linestyle='--', 
           linewidth=2, label=f'{name} mean: {subset.mean():.0f}')

ax.set_title('FICO Score Distribution by Loan Outcome')
ax.set_xlabel('FICO Score')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/03_fico_distribution.png')
plt.close()

# ── Chart 4: Interest Rate Distribution ──────────────────
# Question: Do borrowers who defaulted pay higher interest rates?

fig, ax = plt.subplots(figsize=(8, 5))

for label, colour, name in [(0, '#2ECC71', 'Fully Paid'), 
                             (1, '#E74C3C', 'Default')]:
    subset = df[df['default'] == label]['int_rate']
    ax.hist(subset, bins=40, alpha=0.6, label=name, color=colour, density=True)
    ax.axvline(subset.mean(), color=colour, linestyle='--', 
           linewidth=2, label=f'{name} mean: {subset.mean():.1f}%')
    
ax.set_title('Interest Rate Distribution by Loan Outcome')
ax.set_xlabel('Interest Rate (%)')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/04_interest_rate_distribution.png')
plt.close()


# ── Chart 5: DTI Distribution ──────────────────
# Question: Do borrowers who defaulted have higher DTI ratios?

fig, ax = plt.subplots(figsize=(8, 5))
for label, colour, name in [(0, '#2ECC71', 'Fully Paid'), 
                             (1, '#E74C3C', 'Default')]:
    subset = df[df['default'] == label]['dti']
    ax.hist(subset, bins=640, alpha=0.6, label=name, color=colour, density=True)
    ax.axvline(subset.mean(), color=colour, linestyle='--', 
           linewidth=2, label=f'{name} mean: {subset.mean():.1f}')
ax.set_xlim(0, 80)
ax.set_title('DTI Distribution by Loan Outcome')
ax.set_xlabel('Debt-to-Income Ratio')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/05_dti_distribution.png')
plt.close()

# ── Chart 6: Default Rate by Loan Purpose ───────────────
# Question: Does default rate vary by loan purpose?

purpose_stats = df.groupby('purpose')['default'].mean().reset_index()
purpose_stats.columns = ['purpose', 'default_rate']
purpose_stats = purpose_stats.sort_values('default_rate')

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='default_rate', y='purpose', data=purpose_stats,
            palette='Reds', ax=ax)
ax.set_title('Default Rate by Loan Purpose')
ax.set_xlabel('Default Rate')
ax.set_ylabel('Loan Purpose')

# Add the percentage on top of each bar
for bar in ax.patches:
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.1%}',
            ha='left', va='center', fontsize=11)
ax.set_xlim(0, purpose_stats['default_rate'].max() * 1.2)

plt.tight_layout()
plt.savefig('outputs/figures/06_default_by_purpose.png')
plt.close()

# ── Chart 7: Default Rate by Number of Delinquencies ───────────────
# Question: Does default rate rise with number of delinquencies in past 2 years?

# delinq_2yrs shows weak positive relationship with default
# (19.6% at 0 delinquencies vs ~24% at 5+)
# Relationship is noisy at higher values due to small sample sizes
# Chart omitted — finding documented in README
delinq_stats = df.groupby('delinq_2yrs')['default'].mean().reset_index()
print(delinq_stats.head(10))
print(df['delinq_2yrs'].value_counts().sort_index().head(10))