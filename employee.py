import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional theme set karna formatting ke liye
sns.set_theme(style="whitegrid")

print("============= STEP 1: DATA LOADING & EXPLORATION =============")
# Raw Employee structures dataset (Curriculum standards)
raw_hr_data = {
    'Employee_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    'Name': ['Amit', 'Priya', 'Rohan', 'Sneha', 'Vikram', 'Ananya', 'Rahul', 'Pooja', 'Deepak', 'Kiran', 'Raj', 'Megha', 'Sanjay', 'Neha', 'Vikram'],
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'IT'],
    'Performance_Score': [85, 92, 78, 65, 95, 88, 60, 72, 90, 82, 55, 89, 74, 68, 95],
    'Tenure_Years': [3, 5, 2, 1, 6, 4, 1, 3, 5, 4, 2, 5, 3, 2, 6],
    'Salary': [50000, 60000, 55000, 45000, 80000, 65000, 42000, 48000, 70000, 58000, 41000, 62000, 53000, 46000, 80000],
    'Missing_Field_Test': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10, 11, 12, np.nan, 14, 15]
}

df = pd.DataFrame(raw_hr_data)

# Console printing pipeline [Curriculum Deliverable 3-5]
print("\n[INFO] First 10 Rows of Dataset:")
print(df.head(10))

print(f"\n[INFO] Dataset Shape: {df.shape}")
print("\n[INFO] Missing Values Count Before Cleaning:")
print(df.isnull().sum())

print("\n[INFO] Basic Statistical Summary:")
print(df.describe())

print("\n============= STEP 2: DATA CLEANING =============")
# Cleaning rules implementation [Curriculum Deliverable 6-10]
df['Missing_Field_Test'] = df['Missing_Field_Test'].fillna(df['Missing_Field_Test'].median())
df = df.drop_duplicates()
df['Employee_ID'] = df['Employee_ID'].astype(str)

print("[CLEANING DOCUMENTATION]:")
print("- Missing field components imputed using Median metrics successfully.")
print("- Duplicates removed. Active framework records locked down.")

print("\n============= STEP 3: DATA ANALYSIS =============")
# Core analysis blocks [Curriculum Deliverable 11-16]
dept_perf = df.groupby('Department')['Performance_Score'].mean().reset_index()
print("\n[ANALYSIS] Average Performance by Department:")
print(dept_perf)

top_performers = df.nlargest(10, 'Performance_Score')
print("\n[ANALYSIS] Top Performers Hierarchy:")
print(top_performers[['Employee_ID', 'Name', 'Department', 'Performance_Score']])

correlation_tenure = df['Tenure_Years'].corr(df['Performance_Score'])
correlation_salary = df['Salary'].corr(df['Performance_Score'])
print(f"\n[ANALYSIS] Tenure vs Performance Correlation: {correlation_tenure:.4f}")
print(f"[ANALYSIS] Salary vs Performance Correlation: {correlation_salary:.4f}")

print("\n============= STEP 4: VISUALIZATIONS GENERATION =============")
print("[SYSTEM] .py scripts will launch graph windows. Close each window to see the next chart.")

# Chart 1: Performance Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Performance_Score'], kde=True, color='purple')
plt.title('1. Performance Score Distribution')
plt.tight_layout()
plt.show()
plt.close()

# Chart 2: Department-wise Avg Performance
plt.figure(figsize=(6, 4))
sns.barplot(data=dept_perf, x='Department', y='Performance_Score', palette='muted')
plt.title('2. Department Average Performance')
plt.tight_layout()
plt.show()
plt.close()

# Chart 3: Salary vs Performance (Scatter Plot)
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Salary', y='Performance_Score', hue='Department', s=100)
plt.title('3. Salary vs Performance Metrics')
plt.tight_layout()
plt.show()
plt.close()

# Chart 4: Performance by Tenure
plt.figure(figsize=(6, 4))
sns.lineplot(data=df, x='Tenure_Years', y='Performance_Score', marker='o', color='orange', errorbar=None)
plt.title('4. Performance vs Experience Tenure')
plt.tight_layout()
plt.show()
plt.close()

# Chart 5: Correlation Matrix Heatmap
plt.figure(figsize=(5, 4))
corr_matrix = df[['Performance_Score', 'Tenure_Years', 'Salary']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('5. System Correlation Matrix')
plt.tight_layout()
plt.show()
plt.close()

print("\n============= STEP 5: HR EXECUTIVE REPORT =============")