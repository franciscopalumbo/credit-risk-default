import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv('data/loans_clean.csv')

# Ordinally encode 'grade' (A=1, B=2, ..., G=7)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade'] = df['grade'].map(grade_map)

# One-hot encode 'home_ownership' and 'purpose'
df = pd.get_dummies(df, columns=['home_ownership', 'purpose'], drop_first=True)

# Separate features from target
X = df.drop('default', axis=1)
y = df['default']

# Split into training and test sets
# stratify=y ensures both sets have the same default rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for logistic regression
scaler = StandardScaler()

# Fit on training data only, then transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Model 1: Logistic Regression ──────────────────────
# class_weight='balanced' tells the model to pay more attention
# to the minority class (defaulters are worth 4x as much: 80/20 split) 
# to improve recall
lr_model = LogisticRegression(class_weight='balanced', 
                               max_iter=1000, 
                               random_state=42)

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)
lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]

# ── Model 2: Random Forest ─────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# ── Feature Importance (Random Forest) ─────────────────
# Even though LR performs better, RF gives us feature importance
feature_names = X.columns.tolist()
importances = rf_model.feature_importances_

# Create a sorted dataframe
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=feat_imp.head(10), palette='viridis', ax=ax)
ax.set_title('Top 10 Feature Importances (Random Forest)')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')

for bar in ax.patches:
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.1%}',
            ha='left', va='center', fontsize=11)
ax.set_xlim(0, feat_imp['importance'].max() * 1.2)

plt.tight_layout()
plt.savefig('outputs/figures/07_feature_importance.png')
plt.close()

# ── ROC Curve ──────────────────────────────────────────
# Question: How well does each model separate defaulters 
# from non-defaulters across all thresholds?

fig, ax = plt.subplots(figsize=(7, 6))

for name, probs, color in [
    ('Logistic Regression', lr_probabilities, '#3498DB'),
    ('Random Forest', rf_probabilities, '#E74C3C')
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name} (AUC = {auc:.3f})')

# Random baseline - a useless model scores 0.5
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, 
        label='Random Classifier (AUC = 0.500)')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curves — Model Comparison')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/figures/08_roc_curves.png')
plt.close()

# ── Confusion Matrix ───────────────────────────────────
# Question: How do the models perform at a specific threshold?

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, name, preds in zip(axes, 
                             ['Logistic Regression', 'Random Forest'], 
                             [lr_predictions, rf_predictions]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, vmin=0, vmax=215000)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['Non-Default', 'Default'])
    ax.set_yticklabels(['Non-Default', 'Default'])

plt.tight_layout()
plt.savefig('outputs/figures/09_confusion_matrices.png')
plt.close()