# ==============================================================================
# Fundamentals of Data Science - Case Study 4: Student Performance & Dropout
# Author: Alapati Nagesh (Roll No: 160124733160)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# PART A: Data Ingestion & Preprocessing
file_path = "Studentdata.csv"
df = pd.read_csv(file_path, sep=';')
df.columns = df.columns.str.strip().str.replace('\t', '')

print("[INFO] Dataset shape:", df.shape)
print("[INFO] Missing values total:", df.isnull().sum().sum())
print("[INFO] Duplicate rows total:", df.duplicated().sum())

# PART B: Exploratory Data Analysis & Visualizations
target_counts = df['Target'].value_counts()
target_pcts = df['Target'].value_counts(normalize=True) * 100

print("\n--- Target Outcome Distribution ---")
for k in target_counts.index:
    print(f"{k}: {target_counts[k]} ({target_pcts[k]:.2f}%)")

palette = {'Graduate': '#2b7bba', 'Dropout': '#d9534f', 'Enrolled': '#f0ad4e'}

# Plot 1: Target Distribution
plt.figure(figsize=(8, 4.5))
sns.countplot(data=df, x='Target', order=['Graduate', 'Dropout', 'Enrolled'], palette=palette)
plt.title("Figure 1: Target Class Distribution (Academic Outcome)", fontsize=13, weight="bold")
plt.xlabel("Student Final Outcome")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("cs4_fig1_target_distribution.png", dpi=300)
plt.close()

# Plot 2: Numerical Distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df['Age at enrollment'], bins=25, kde=True, ax=axes[0], color='#2b5c8f')
axes[0].set_title("Age at Enrollment Distribution", weight="bold")
axes[0].set_xlabel("Age at Enrollment (Years)")
axes[0].set_ylabel("Count of Students")

sns.histplot(df['Admission grade'], bins=25, kde=True, ax=axes[1], color='#388e3c')
axes[1].set_title("Admission Grade Distribution", weight="bold")
axes[1].set_xlabel("Admission Grade (0 - 200 Scale)")
axes[1].set_ylabel("Count of Students")
plt.tight_layout()
plt.savefig("cs4_fig2_numerical_distributions.png", dpi=300)
plt.close()

# Plot 3: Boxplot of Admission Grades
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x='Target', y='Admission grade', order=['Graduate', 'Dropout', 'Enrolled'], palette=palette)
plt.title("Figure 3: Admission Grade Comparison Across Outcome Classes", fontsize=13, weight="bold")
plt.xlabel("Student Academic Outcome")
plt.ylabel("Admission Grade (0 - 200 Scale)")
plt.tight_layout()
plt.savefig("cs4_fig3_admission_grade_boxplot.png", dpi=300)
plt.close()

# Plot 4: First-Semester Grade vs Outcome
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (grade)', order=['Graduate', 'Dropout', 'Enrolled'], palette=palette)
plt.title("Figure 4: 1st Semester Grade Across Outcome Classes", fontsize=13, weight="bold")
plt.xlabel("Student Academic Outcome")
plt.ylabel("Curricular Units 1st Sem Grade (0 - 20 Scale)")
plt.tight_layout()
plt.savefig("cs4_fig4_first_sem_grades.png", dpi=300)
plt.close()

# PART C: Hypothesis Testing
grad_grades = df[df['Target'] == 'Graduate']['Curricular units 1st sem (grade)']
drop_grades = df[df['Target'] == 'Dropout']['Curricular units 1st sem (grade)']
enr_grades = df[df['Target'] == 'Enrolled']['Curricular units 1st sem (grade)']

f_stat, p_val = stats.f_oneway(grad_grades, drop_grades, enr_grades)
kw_stat, kw_pval = stats.kruskal(grad_grades, drop_grades, enr_grades)

print("\n--- Hypothesis Testing: First-Semester Academic Performance ---")
print(f"One-Way ANOVA F-statistic : {f_stat:.4f}")
print(f"One-Way ANOVA p-value     : {p_val:.4e}")
print(f"Kruskal-Wallis H-statistic: {kw_stat:.4f}")
print(f"Kruskal-Wallis p-value    : {kw_pval:.4e}")
if p_val < 0.05:
    print("[DECISION] Reject Null Hypothesis (H0) at alpha = 0.05. Statistically significant difference exists.")

# PART D: Multiclass Logistic Regression
X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

print("\n--- Model Performance Evaluation ---")
acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Plot 5: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred, labels=['Dropout', 'Enrolled', 'Graduate'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dropout', 'Enrolled', 'Graduate'],
            yticklabels=['Dropout', 'Enrolled', 'Graduate'])
plt.title("Figure 5: Confusion Matrix (Multiclass Logistic Regression)", fontsize=12, weight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("Actual True Label")
plt.tight_layout()
plt.savefig("cs4_fig5_confusion_matrix.png", dpi=300)
plt.close()

print("[INFO] Execution finished successfully. All figures saved.")
