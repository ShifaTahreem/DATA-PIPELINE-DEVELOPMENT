DATA-PIPELINE-DEVELOPMENT

COMPANY:CODECH IT SOLUTIONS

INTERN ID:

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

PROJECT DESCRIPTION:

ETL Data Pipeline using Pandas and Scikit-learn

This project was developed as part of my internship with CodTech IT Solutions. The goal was to understand and implement a foundational ETL (Extract, Transform, Load) pipeline using Python and key data science libraries such as Pandas and Scikit-learn. This pipeline showcases essential concepts in data preprocessing, transformation, and automated data handling.

PROJECT OBJECTIVE:

The objective of this project was to simulate a real-world data engineering scenario where raw data needs to be cleaned, processed, and stored in a usable format for analysis or machine learning. The ETL process is crucial in almost all data-related workflows — whether in data science, machine learning, or business intelligence. Through this task, I aimed to gain hands-on experience with the key steps involved in data processing:

Extracting raw data from a source file (CSV format).

Transforming the data by handling missing values, encoding categorical variables, and scaling numerical features.

Loading the final cleaned data into a new CSV file for future use.

TOOLS AND TECHNOLOGIES USED:

Python – Main programming language used for scripting the ETL pipeline.

Pandas – Used for data reading, manipulation, and cleaning.

Scikit-learn (sklearn) – Used for preprocessing operations like scaling and encoding.

VS Code – IDE used for writing and running the Python script.

Git & GitHub – Version control and hosting the project code online.

ETL PROCESS BREAKDOWN:
1. Extraction

The data is extracted from a CSV file named raw_data.csv, which contains records with the following columns:

Name

Age

Gender

Salary

Some rows have missing values, which is a common situation in real datasets.

2. Transformation

This is the core part of the pipeline. The following steps are performed:

Missing Value Handling:

For numerical columns like Age and Salary, missing values are filled with the mean of the column.

Categorical Encoding:

The Gender column is converted to numerical values using label encoding (Male = 1, Female = 0).

Feature Scaling:

Columns such as Age and Salary are scaled using StandardScaler from Scikit-learn, which normalizes the values to have a mean of 0 and a standard deviation of 1. This is essential for many machine learning algorithms to perform effectively.

3. Loading

The transformed dataset is saved to a new file named cleaned_data.csv. This cleaned version can now be directly used for further data analysis or fed into a machine learning model.

WHAT I LEARNED:

This project was an excellent learning opportunity to understand how real-world data is rarely clean and how it requires multiple steps to make it usable. I gained confidence in:

Reading and inspecting data using Pandas.

Applying basic but essential data cleaning techniques.

Transforming and preparing data using Scikit-learn tools.

Automating a complete ETL process in Python.

Using Git and GitHub for version control and collaboration.

OUTCOME:

The ETL process runs successfully end-to-end and prints the following messages in the terminal:

Displays the raw and cleaned data for comparison.

Confirms that the final CSV file has been created.

The project files are well-commented and stored in a public GitHub repository for review. This project marks an important step in my data science and engineering journey, and I look forward to building more advanced pipelines in the future.

OUTPUT:

<img width="960" height="540" alt="Image" src="https://github.com/user-attachments/assets/dead76d7-6def-46e9-a244-f5f370937086" />


