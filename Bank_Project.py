import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("bank.csv")
print(df)

print('-------------------------------------------------------------------------------------')

# Display column names
print(df.columns)

print('-------------------------------------------------------------------------------------')

# Rename columns
df.rename(columns={'marital': 'Marital_Status',
                    'pdays': 'Days_Since_Last_Contact',
                    'poutcome': 'Previous_Outcome',
                    'y': 'Customer_Subscription'}, inplace=True)
print(df.columns)

print('-------------------------------------------------------------------------------------')

# Display dataset information
print(df.info())

print('-------------------------------------------------------------------------------------')

# Display statistical summary
print(df.describe())

print('-------------------------------------------------------------------------------------')

# Check data types
print(df.dtypes)

print('-------------------------------------------------------------------------------------')

# Convert categorical columns to category dtype
categorical_columns = ['job', 'Marital_Status', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'Previous_Outcome', 'Customer_Subscription']
for col in categorical_columns:
    df[col] = df[col].astype('category')
print("Data Types after conversion:\n", df.dtypes)

print('-------------------------------------------------------------------------------------')

# Check for duplicate rows
df_duplicated = df.duplicated()
print(df.duplicated().sum())

print('-------------------------------------------------------------------------------------')

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

print('-------------------------------------------------------------------------------------')

# Check for missing values
print(df.isnull().sum())

print('-------------------------------------------------------------------------------------')

# Fill missing values in 'Balance' column with mean
df['balance'] = df['balance'].fillna(df['balance'].mean())
print(df['balance'].isna().sum())

print('-------------------------------------------------------------------------------------')

# Fill missing values in 'Campaign' column with mean
df['campaign'] = df['campaign'].fillna(df['campaign'].mode()[0])
print(df['campaign'].isnull().sum())

print('-------------------------------------------------------------------------------------')

# Final missing values check
print(df.isnull().sum())

print('-------------------------------------------------------------------------------------')

# Final dataset information
print(df.info())

print('-------------------------------------------------------------------------------------')

# 1. How many married individuals are unemployed?

# Filter the dataset for married individuals who are unemployed
married_unemployed = df[(df['Marital_Status'] == 'married') & (df['job'] == 'unknown')]

# Count the number of married unemployed individuals
count_married_unemployed = married_unemployed.shape[0]
print(f"Number of married unemployed individuals: {count_married_unemployed}")

# Visualization: Pie chart
total_count = df.shape[0]
sizes = [count_married_unemployed, total_count - count_married_unemployed]

plt.pie(sizes, labels=['Married Unemployed', 'Other'], autopct='%1.1f%%', startangle=90,
        colors=['#ff6666', '#66b3ff'], wedgeprops={'edgecolor': 'black'})
plt.title('Proportion of Married Unemployed Individuals', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. What is the average age of individuals who subscribed to a term deposit versus those who did not?

# Calculate the average age for both groups: Subscribed and Not Subscribed
avg_age_subscribed = df[df['Customer_Subscription'] == 'yes']['age'].mean()
avg_age_not_subscribed = df[df['Customer_Subscription'] == 'no']['age'].mean()

# Data for pie chart
labels = ['Subscribed', 'Not Subscribed']
average_ages = [avg_age_subscribed, avg_age_not_subscribed]

# Create a pie chart
plt.figure(figsize=(10,6))
plt.pie(average_ages, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])

# Adding title
plt.title('Average Age of Individuals Who Subscribed vs. Did Not Subscribe to a Term Deposit', fontsize=16, fontweight='bold')

# Show the plot
plt.tight_layout()
plt.show()

# # 3. How does the average balance vary across different contact months?

# Group by 'Month' and calculate the average balance for each month
avg_balance_per_month = df.groupby('month')['balance'].mean()

# Visualization: Horizontal bar chart for average balance across months with similar colors
plt.figure(figsize=(10,6))
avg_balance_per_month.plot(kind='barh', color=['#66b3ff', '#ff9999', '#66b3ff', '#ff9999', '#66b3ff', '#ff9999', '#66b3ff', '#ff9999', '#66b3ff', '#ff9999', '#66b3ff', '#ff9999'])

# Adding titles and labels
plt.title('Average Balance Across Different Contact Months', fontsize=16, fontweight='bold')
plt.ylabel('Month', fontsize=12)
plt.xlabel('Average Balance', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# # 4. How does the average duration of customer calls vary by month?

# Group by 'Month' and calculate the average duration of customer calls for each month
avg_duration_per_month = df.groupby('month')['duration'].mean()

# Visualization: Line chart for average call duration across months
plt.figure(figsize=(10,6))
avg_duration_per_month.plot(kind='line', marker='o', color='#66b3ff', linestyle='-', linewidth=2)

# Adding titles and labels
plt.title('Average Duration of Customer Calls Across Different Months', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Duration (seconds)', fontsize=12)

# Show the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# # 5. What is the distribution of marital status (married/single/divorced) across different education levels?

# Group by 'Education' and 'Marital_Status' to get the count of each marital status for each education level
education_marital_counts = df.groupby(['education', 'Marital_Status']).size().unstack(fill_value=0)

# Plotting the stacked bar chart
education_marital_counts.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999', '#ffcc66'], figsize=(10,6))

# Adding titles and labels
plt.title('Distribution of Marital Status Across Different Education Levels', fontsize=16, fontweight='bold')
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Number of Individuals', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Univariate Analysis
numerical_cols = ['age', 'balance', 'duration', 'campaign', 'previous']
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='job', y='balance', data=df, ci=None)
plt.title("Average Balance by Job Type")
plt.xticks(rotation=45)
plt.show()

# Customer Segmentation
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan', y='balance', data=df)
plt.title("Balance Distribution by Loan Status")
plt.show()

# Correlation Analysis: Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Final Insights and Reporting

print("\nFinal Insights:\n")
print("1. Number of married unemployed individuals:", count_married_unemployed)
print('-------------------------------------------------------------------------------------')
print("2. Average age of individuals who subscribed to term deposit:", avg_age_subscribed)
print('-------------------------------------------------------------------------------------')
print("3. Average balance across contact months:\n", avg_balance_per_month)
print('-------------------------------------------------------------------------------------')
print("4. Average call duration across months:\n", avg_duration_per_month)
print('-------------------------------------------------------------------------------------')
print("5. Distribution of marital status across education levels:\n", education_marital_counts)
