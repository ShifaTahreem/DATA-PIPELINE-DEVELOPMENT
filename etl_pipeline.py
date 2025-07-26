# Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# STEP 1: EXTRACT - Load the raw dataset
df = pd.read_csv('raw_data.csv')
print("Raw Data:")
print(df)

# STEP 2: TRANSFORM

# 2.1 Handle missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# 2.2 Convert text labels (Gender) into numbers
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  # Male=1, Female=0

# 2.3 Scale Age and Salary for better modeling later
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\nCleaned & Transformed Data:")
print(df)

# STEP 3: LOAD - Save the cleaned data to a new CSV file
df.to_csv('cleaned_data.csv', index=False)
print("\nETL completed successfully. File 'cleaned_data.csv' created.")
