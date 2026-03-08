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

fig, ax = plt.subplots(figsize=(7, 4))
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