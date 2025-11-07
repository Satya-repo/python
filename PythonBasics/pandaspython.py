# ==================================================================================
# COMPREHENSIVE PANDAS DATA CLEANING & FEATURE ENGINEERING - REAL WORLD USE CASE
# ==================================================================================
# 
# PURPOSE: This script demonstrates how to use pandas for data cleaning and
# feature engineering using a real-world e-commerce orders dataset.
#
# USE CASE: Analyzing e-commerce order data to prepare it for business insights
# and machine learning models.
#
# TARGET AUDIENCE: Beginners to pandas and data science
#
# WHAT YOU'LL LEARN:
#   1. Loading and inspecting data
#   2. Handling missing values and duplicates
#   3. Cleaning string data
#   4. Feature engineering (creating new features from existing data)
#   5. Data transformations for ML
#   6. Exporting cleaned data
#
# ==================================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

print("="*80)
print("PANDAS DATA CLEANING & FEATURE ENGINEERING - REAL WORLD TUTORIAL")
print("="*80)
print("\n📊 Use Case: E-commerce Order Analysis & ML Preparation")
print("📁 Dataset: orders.csv (E-commerce order records)")
print("🎯 Goal: Clean data and create features for business insights & ML models")
print("\n" + "="*80)

# ============================================================================
# STEP 1: LOADING DATA FROM CSV FILE
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA FROM CSV FILE")
print("="*80)

# WHAT IS pd.read_csv()?
# - Reads a CSV (Comma Separated Values) file into a pandas DataFrame
# - DataFrame is like a spreadsheet in Python (rows and columns)
# - Think of it as Excel table in code form

# SYNTAX: pd.read_csv('file_path', parse_dates=['column_name'])
# - file_path: Location of your CSV file
# - parse_dates: Automatically converts date strings to datetime objects
#                This makes date operations easier later

orders = pd.read_csv(
    '/Users/satya/PythonLearn/PythonBasics/orders.csv',
    parse_dates=['OrderDate']  # Convert OrderDate from string to date type
)

print("\n✅ Data loaded successfully!")
print(f"📊 Dataset shape: {orders.shape[0]} rows × {orders.shape[1]} columns")
print(f"   (Rows = individual orders, Columns = data fields)")

# ============================================================================
# STEP 2: INITIAL DATA INSPECTION - Understanding Your Data
# ============================================================================

print("\n" + "="*80)
print("STEP 2: INITIAL DATA INSPECTION - Understanding Your Data")
print("="*80)

# .head() - Shows first 5 rows (by default)
# WHY: Quick preview to see what data looks like
print("\n1️⃣ FIRST 5 ROWS (Preview of data):")
print("-" * 80)
print(orders.head())

# .info() - Shows data types, missing values, and memory usage
# WHY: Understand data structure and identify problems early
print("\n2️⃣ DATA INFORMATION (Types, missing values, memory):")
print("-" * 80)
orders.info()

# .describe() - Statistical summary for numeric columns
# WHY: Get quick statistics (mean, min, max, etc.) to spot outliers
print("\n3️⃣ STATISTICAL SUMMARY (For numeric columns):")
print("-" * 80)
print(orders.describe())

# .dtypes - Shows data type of each column
# WHY: Know what type of data each column contains
print("\n4️⃣ DATA TYPES (What type is each column?):")
print("-" * 80)
for col, dtype in orders.dtypes.items():
    print(f"   {col:20s} → {str(dtype)}")

# ============================================================================
# STEP 3: DATA CLEANING - Handling Missing Values
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DATA CLEANING - Handling Missing Values")
print("="*80)

# WHAT ARE MISSING VALUES?
# - Empty cells in your data (NaN = Not a Number)
# - Real data often has missing values that need to be handled
# - Different strategies: fill with mean, remove rows, or impute

# .isna() or .isnull() - Finds missing values
# Returns True where value is missing, False otherwise
print("\n1️⃣ CHECKING FOR MISSING VALUES:")
print("-" * 80)
missing_count = orders.isna().sum()
missing_percent = (missing_count / len(orders)) * 100

missing_info = pd.DataFrame({
    'Column': missing_count.index,
    'Missing Count': missing_count.values,
    'Missing %': missing_percent.values
})

# Only show columns with missing values
missing_info = missing_info[missing_info['Missing Count'] > 0]

if len(missing_info) > 0:
    print("⚠️  Found missing values:")
    print(missing_info.to_string(index=False))
else:
    print("✅ No missing values found! Data is complete.")

# HANDLING MISSING VALUES - Common strategies:
# Strategy 1: Fill with a default value
#   orders['Quantity'].fillna(0)  # Fill missing quantities with 0
#
# Strategy 2: Fill with mean/median
#   orders['Price'].fillna(orders['Price'].mean())  # Fill with average
#
# Strategy 3: Remove rows with missing values
#   orders.dropna()  # Remove any row with missing data

# For our dataset, we'll check and handle any missing values
if orders['Quantity'].isna().any():
    orders['Quantity'] = orders['Quantity'].fillna(0)
    print("\n✅ Filled missing Quantity values with 0")

if orders['Price'].isna().any():
    orders['Price'] = orders['Price'].fillna(orders['Price'].mean())
    print("✅ Filled missing Price values with mean")

# ============================================================================
# STEP 4: DATA TYPE CONVERSION - Ensuring Correct Data Types
# ============================================================================

print("\n" + "="*80)
print("STEP 4: DATA TYPE CONVERSION - Ensuring Correct Data Types")
print("="*80)

# WHY CONVERT DATA TYPES?
# - String columns should be 'string' type (not 'object')
# - Numeric columns should be int/float
# - Correct types = better performance and fewer errors

print("\n1️⃣ CONVERTING STRING COLUMNS:")
print("-" * 80)

# .astype('string') - Converts column to string type
# WHY: More efficient than 'object' type, better for string operations
string_columns = ['CustomerName', 'Product', 'Category', 'Country']
for col in string_columns:
    if col in orders.columns:
        orders[col] = orders[col].astype('string')
        print(f"   ✅ {col} → string type")

print("\n2️⃣ ENSURING NUMERIC TYPES:")
print("-" * 80)

# Ensure numeric columns are correct type
if orders['Quantity'].dtype != 'int64':
    orders['Quantity'] = orders['Quantity'].astype('int64')
    print("   ✅ Quantity → integer type")

if orders['Price'].dtype != 'float64':
    orders['Price'] = orders['Price'].astype('float64')
    print("   ✅ Price → float type")

# ============================================================================
# STEP 5: STRING DATA CLEANING - Fixing Text Data Issues
# ============================================================================

print("\n" + "="*80)
print("STEP 5: STRING DATA CLEANING - Fixing Text Data Issues")
print("="*80)

# COMMON TEXT DATA PROBLEMS:
# - Extra spaces (leading/trailing whitespace)
# - Inconsistent capitalization
# - Typos or variations in spelling

print("\n1️⃣ REMOVING WHITESPACE (Leading/trailing spaces):")
print("-" * 80)

# .str.strip() - Removes spaces from beginning and end
# WHY: "John Smith" vs "John Smith " are different strings but same person
for col in string_columns:
    if col in orders.columns:
        orders[col] = orders[col].str.strip()
        print(f"   ✅ Cleaned {col} (removed extra spaces)")

print("\n2️⃣ STANDARDIZING CASE (Converting to lowercase):")
print("-" * 80)

# .str.lower() - Converts all text to lowercase
# WHY: "Yes" and "yes" are different but mean the same thing
if 'Shipped' in orders.columns:
    orders['Shipped'] = orders['Shipped'].str.lower()
    print("   ✅ Converted Shipped column to lowercase")

print("\n3️⃣ CONVERTING TO BOOLEAN (True/False):")
print("-" * 80)

# .map(dictionary) - Maps values using a dictionary
# WHY: ML models work better with True/False than "yes"/"no"
if 'Shipped' in orders.columns:
    shipped_map = {'yes': True, 'no': False}
    orders['Shipped'] = orders['Shipped'].map(shipped_map)
    print("   ✅ Converted Shipped to boolean (True/False)")
    print(f"      Shipped status: {orders['Shipped'].value_counts().to_dict()}")

# ============================================================================
# STEP 6: REMOVING DUPLICATES - Eliminating Repeated Records
# ============================================================================

print("\n" + "="*80)
print("STEP 6: REMOVING DUPLICATES - Eliminating Repeated Records")
print("="*80)

# WHAT ARE DUPLICATES?
# - Identical rows that appear multiple times
# - Can skew analysis results

# .duplicated() - Finds duplicate rows
# Returns True for rows that are duplicates
print("\n1️⃣ CHECKING FOR DUPLICATES:")
print("-" * 80)
duplicate_count = orders.duplicated().sum()
print(f"   Found {duplicate_count} duplicate row(s)")

# .drop_duplicates() - Removes duplicate rows
# WHY: Keep only unique records
if duplicate_count > 0:
    orders = orders.drop_duplicates()
    print(f"   ✅ Removed duplicates. New shape: {orders.shape}")
else:
    print("   ✅ No duplicates found")

# ============================================================================
# STEP 7: FEATURE ENGINEERING - Creating New Features from Existing Data
# ============================================================================

print("\n" + "="*80)
print("STEP 7: FEATURE ENGINEERING - Creating New Features")
print("="*80)

# WHAT IS FEATURE ENGINEERING?
# - Creating new columns from existing data
# - These new features help ML models make better predictions
# - Example: Calculate total amount from quantity × price

print("\n1️⃣ CALCULATED FEATURES (Derived from existing columns):")
print("-" * 80)

# Create TotalAmount = Quantity × Price
# WHY: ML models need this calculated value, not just quantity and price separately
orders['TotalAmount'] = orders['Quantity'] * orders['Price']
print("   ✅ Created TotalAmount = Quantity × Price")
print(f"      Example: {orders[['Quantity', 'Price', 'TotalAmount']].head(3).to_string(index=False)}")

print("\n2️⃣ DATE-BASED FEATURES (Extracting information from dates):")
print("-" * 80)

# .dt accessor - Allows datetime operations
# WHY: Extract year, month, day, etc. from dates for time-based analysis
orders['OrderYear'] = orders['OrderDate'].dt.year
orders['OrderMonth'] = orders['OrderDate'].dt.month
orders['OrderDay'] = orders['OrderDate'].dt.day
orders['OrderDayOfWeek'] = orders['OrderDate'].dt.dayofweek  # 0=Monday, 6=Sunday
orders['OrderQuarter'] = orders['OrderDate'].dt.quarter
orders['OrderWeekday'] = orders['OrderDate'].dt.day_name()  # Monday, Tuesday, etc.

print("   ✅ Extracted date features:")
print("      - OrderYear: Year of order")
print("      - OrderMonth: Month (1-12)")
print("      - OrderDay: Day of month")
print("      - OrderDayOfWeek: Day of week (0=Monday, 6=Sunday)")
print("      - OrderQuarter: Quarter (1-4)")
print("      - OrderWeekday: Day name (Monday, Tuesday, etc.)")

# Show example
date_features = orders[['OrderDate', 'OrderYear', 'OrderMonth', 'OrderDayOfWeek', 'OrderWeekday']].head(3)
print(f"\n      Example values:")
print(f"      {date_features.to_string(index=False)}")

print("\n3️⃣ CUSTOMER-BASED FEATURES (Aggregations per customer):")
print("-" * 80)

# .groupby() - Groups rows by a column value
# .transform() - Applies function but keeps original DataFrame shape
# WHY: Creates features that show customer behavior patterns

# Count total orders per customer
orders['CustomerOrderCount'] = orders.groupby('CustomerName')['OrderID'].transform('count')
print("   ✅ Created CustomerOrderCount (total orders per customer)")

# Calculate average order value per customer
orders['CustomerAvgOrderValue'] = orders.groupby('CustomerName')['TotalAmount'].transform('mean')
print("   ✅ Created CustomerAvgOrderValue (average order value per customer)")

# Calculate total spending per customer
orders['CustomerTotalSpent'] = orders.groupby('CustomerName')['TotalAmount'].transform('sum')
print("   ✅ Created CustomerTotalSpent (total spending per customer)")

# Show example
customer_features = orders[['CustomerName', 'CustomerOrderCount', 
                           'CustomerAvgOrderValue', 'CustomerTotalSpent']].head(5)
print(f"\n      Example customer features:")
print(f"      {customer_features.to_string(index=False)}")

print("\n4️⃣ CATEGORY-BASED FEATURES (Product category insights):")
print("-" * 80)

# Average price per category
orders['CategoryAvgPrice'] = orders.groupby('Category')['Price'].transform('mean')
print("   ✅ Created CategoryAvgPrice (average price in category)")

# Total orders per category
orders['CategoryOrderCount'] = orders.groupby('Category')['OrderID'].transform('count')
print("   ✅ Created CategoryOrderCount (total orders in category)")

# ============================================================================
# STEP 8: CATEGORICAL ENCODING - Converting Categories to Numbers for ML
# ============================================================================

print("\n" + "="*80)
print("STEP 8: CATEGORICAL ENCODING - Converting Categories to Numbers")
print("="*80)

# WHY ENCODE CATEGORIES?
# - ML models work with numbers, not text
# - "Electronics", "Furniture", "Stationery" need to become numbers
# - One-hot encoding: Creates binary columns (0 or 1) for each category

print("\n1️⃣ ONE-HOT ENCODING (Creating binary columns for each category):")
print("-" * 80)

# pd.get_dummies() - Creates binary columns for each unique category value
# Example: Category "Electronics" → Category_Electronics = 1, others = 0
category_encoded = pd.get_dummies(orders['Category'], prefix='Category')
print(f"   ✅ Encoded Category column into {category_encoded.shape[1]} binary columns")
print(f"      Columns created: {list(category_encoded.columns)}")

country_encoded = pd.get_dummies(orders['Country'], prefix='Country')
print(f"   ✅ Encoded Country column into {country_encoded.shape[1]} binary columns")
print(f"      Columns created (first 5): {list(country_encoded.columns[:5])}...")

# Combine original data with encoded columns
# pd.concat() - Combines DataFrames horizontally (axis=1) or vertically (axis=0)
orders_ml = pd.concat([
    orders.drop(columns=['Category', 'Country']),  # Remove original categorical columns
    category_encoded,
    country_encoded
], axis=1)

print(f"\n   ✅ Combined DataFrame:")
print(f"      Original shape: {orders.shape}")
print(f"      ML-ready shape: {orders_ml.shape}")
print(f"      Added {orders_ml.shape[1] - orders.shape[1]} new feature columns")

# ============================================================================
# STEP 9: DATA FILTERING - Selecting Specific Subsets of Data
# ============================================================================

print("\n" + "="*80)
print("STEP 9: DATA FILTERING - Selecting Specific Subsets")
print("="*80)

# WHY FILTER DATA?
# - Focus on specific segments (e.g., high-value orders)
# - Analyze specific time periods
# - Extract data for specific business questions

print("\n1️⃣ BOOLEAN INDEXING (Filtering rows by conditions):")
print("-" * 80)

# Syntax: df[df['column'] condition]
# Filters rows where condition is True
shipped_orders = orders[orders['Shipped'] == True]
print(f"   ✅ Shipped orders: {len(shipped_orders)} out of {len(orders)}")
print(f"      Percentage: {len(shipped_orders)/len(orders)*100:.1f}%")

high_value_orders = orders[orders['TotalAmount'] > 200]
print(f"   ✅ High-value orders (>$200): {len(high_value_orders)}")
print(f"      Average value: ${high_value_orders['TotalAmount'].mean():.2f}")

# Multiple conditions: use & (and), | (or), ~ (not)
# IMPORTANT: Put each condition in parentheses
electronics_high_value = orders[(orders['Category'] == 'Electronics') & 
                               (orders['TotalAmount'] > 100)]
print(f"   ✅ Electronics orders >$100: {len(electronics_high_value)}")

print("\n2️⃣ QUERY METHOD (SQL-like filtering - more readable):")
print("-" * 80)

# .query() - SQL-like syntax for filtering
# WHY: More readable for complex conditions
recent_shipped = orders.query('Shipped == True and TotalAmount > 0 and OrderYear == 2024')
print(f"   ✅ Recent shipped orders (2024): {len(recent_shipped)}")

# ============================================================================
# STEP 10: AGGREGATIONS & SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: AGGREGATIONS & SUMMARY STATISTICS")
print("="*80)

print("\n1️⃣ GROUPBY AGGREGATIONS (Statistics by category):")
print("-" * 80)

# .groupby().agg() - Groups data and applies multiple functions
category_stats = orders.groupby('Category').agg({
    'TotalAmount': ['count', 'sum', 'mean', 'max'],
    'Quantity': 'sum',
    'Price': 'mean'
}).round(2)

# Flatten column names for readability
category_stats.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 
                         'Max_Order_Value', 'Total_Quantity', 'Avg_Price']
print(category_stats)

print("\n2️⃣ PIVOT TABLE (Cross-tabulation):")
print("-" * 80)

# pd.pivot_table() - Creates cross-tabulation (like Excel pivot table)
# Shows revenue by country and month
pivot_revenue = pd.pivot_table(
    orders,
    index='Country',
    columns='OrderMonth',
    values='TotalAmount',
    aggfunc='sum',
    fill_value=0
)

print("   Revenue by Country and Month (first 5 countries):")
print(pivot_revenue.head())

# ============================================================================
# STEP 11: OUTLIER DETECTION - Finding Unusual Values
# ============================================================================

print("\n" + "="*80)
print("STEP 11: OUTLIER DETECTION - Finding Unusual Values")
print("="*80)

# WHAT ARE OUTLIERS?
# - Extreme values that are very different from most data
# - Can skew analysis and ML model performance
# - Example: One order of $10,000 when average is $100

# IQR Method (Interquartile Range) - Common outlier detection method
Q1 = orders['TotalAmount'].quantile(0.25)  # 25th percentile
Q3 = orders['TotalAmount'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile Range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n📊 TotalAmount Statistics:")
print(f"   Q1 (25th percentile): ${Q1:.2f}")
print(f"   Q3 (75th percentile): ${Q3:.2f}")
print(f"   IQR (Interquartile Range): ${IQR:.2f}")
print(f"   Lower bound: ${lower_bound:.2f}")
print(f"   Upper bound: ${upper_bound:.2f}")

outliers = orders[(orders['TotalAmount'] < lower_bound) | 
                 (orders['TotalAmount'] > upper_bound)]
print(f"\n   ⚠️  Potential outliers: {len(outliers)} orders")
print(f"   Percentage: {len(outliers)/len(orders)*100:.1f}%")

if len(outliers) > 0:
    print(f"\n   Example outliers:")
    print(outliers[['OrderID', 'CustomerName', 'TotalAmount', 'Category']].head())

# ============================================================================
# STEP 12: PREPARING DATA FOR MACHINE LEARNING
# ============================================================================

print("\n" + "="*80)
print("STEP 12: PREPARING DATA FOR MACHINE LEARNING")
print("="*80)

# ML models need:
# - X: Feature matrix (all input variables)
# - y: Target variable (what we want to predict)

print("\n1️⃣ SELECTING FEATURES (X - Input variables):")
print("-" * 80)

# Numeric features (already in correct format)
numeric_features = ['Quantity', 'Price', 'TotalAmount', 'OrderYear', 
                   'OrderMonth', 'OrderDayOfWeek', 'CustomerOrderCount', 
                   'CustomerAvgOrderValue', 'CustomerTotalSpent', 
                   'CategoryAvgPrice', 'CategoryOrderCount']

# Categorical features (one-hot encoded)
categorical_features = [col for col in orders_ml.columns 
                       if col.startswith('Category_') or col.startswith('Country_')]

# Combine all features
X = orders_ml[numeric_features + categorical_features]
print(f"   ✅ Feature matrix (X) shape: {X.shape}")
print(f"      - Numeric features: {len(numeric_features)}")
print(f"      - Categorical features (one-hot): {len(categorical_features)}")
print(f"      - Total features: {len(numeric_features) + len(categorical_features)}")

print("\n2️⃣ SELECTING TARGET (y - What to predict):")
print("-" * 80)

# Example: Predict if order will be shipped (classification problem)
y = orders_ml['Shipped'].astype(int)  # Convert True/False to 1/0
print(f"   ✅ Target variable (y) shape: {y.shape}")
print(f"      Target: Shipped (1 = Yes, 0 = No)")
print(f"      Distribution:")
target_dist = y.value_counts()
for val, count in target_dist.items():
    print(f"         {val} (Shipped={'Yes' if val==1 else 'No'}): {count} orders ({count/len(y)*100:.1f}%)")

print("\n3️⃣ TRAIN/TEST SPLIT (Dividing data for model training):")
print("-" * 80)

# In production, use: from sklearn.model_selection import train_test_split
# For demonstration, we'll do a simple split
train_size = int(0.8 * len(orders_ml))  # 80% for training

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print(f"   ✅ Data split:")
print(f"      Training set: {X_train.shape[0]} samples (80%)")
print(f"      Test set: {X_test.shape[0]} samples (20%)")
print(f"      Features: {X_train.shape[1]}")

# ============================================================================
# STEP 13: EXPORTING CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 13: EXPORTING CLEANED DATA")
print("="*80)

# WHY EXPORT?
# - Save cleaned data for future use
# - Share with team members
# - Use in other tools (Excel, Power BI, etc.)

print("\n1️⃣ SAVING CLEANED DATA:")
print("-" * 80)

# Save as CSV (human-readable, works everywhere)
csv_path = '/Users/satya/PythonLearn/PythonBasics/orders_cleaned.csv'
orders.to_csv(csv_path, index=False)  # index=False prevents saving row numbers
print(f"   ✅ Saved cleaned data to: {csv_path}")
print(f"      Format: CSV (Excel-readable)")
print(f"      Rows: {orders.shape[0]}, Columns: {orders.shape[1]}")

# Save ML-ready features
ml_path = '/Users/satya/PythonLearn/PythonBasics/orders_ml_ready.csv'
X.to_csv(ml_path, index=False)
print(f"   ✅ Saved ML-ready features to: {ml_path}")
print(f"      Format: CSV")
print(f"      Features: {X.shape[1]}")

# Try to save as Parquet (more efficient, preserves data types)
try:
    parquet_path = '/Users/satya/PythonLearn/PythonBasics/orders_cleaned.parquet'
    orders.to_parquet(parquet_path, index=False)
    print(f"   ✅ Saved cleaned data to: {parquet_path}")
    print(f"      Format: Parquet (efficient, preserves data types)")
except Exception as e:
    print(f"   ⚠️  Parquet save skipped: {e} (pyarrow may need to be installed)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL SUMMARY - DATA CLEANING & FEATURE ENGINEERING COMPLETE")
print("="*80)

print(f"""
✅ DATA PROCESSING SUMMARY:

📥 INPUT:
   - File: orders.csv
   - Original shape: {orders.shape[0]} rows × {orders.shape[1]} columns

🔧 DATA CLEANING PERFORMED:
   ✅ Checked and handled missing values
   ✅ Converted data types (string, numeric, datetime)
   ✅ Cleaned string data (removed whitespace, standardized case)
   ✅ Converted boolean columns
   ✅ Removed duplicate records

🏗️ FEATURE ENGINEERING PERFORMED:
   ✅ Created calculated features (TotalAmount)
   ✅ Extracted date features (Year, Month, Day, Quarter, Weekday)
   ✅ Created customer features (Order count, Avg value, Total spent)
   ✅ Created category features (Avg price, Order count)
   ✅ Encoded categorical variables (One-hot encoding)

📤 OUTPUT:
   - Cleaned data: {orders.shape[0]} rows × {orders.shape[1]} columns
   - ML-ready features: {X.shape[0]} samples × {X.shape[1]} features
   - Files saved: orders_cleaned.csv, orders_ml_ready.csv

🎯 DATA READY FOR:
   ✅ Business analysis and reporting
   ✅ Machine learning model training
   ✅ Data visualization
   ✅ Further data processing

📚 KEY PANDAS FUNCTIONS USED:
   1. pd.read_csv()           - Load data
   2. df.head(), df.info()     - Inspect data
   3. df.isna(), df.fillna()   - Handle missing values
   4. df.astype()              - Convert data types
   5. df.str.strip()           - Clean strings
   6. df.drop_duplicates()     - Remove duplicates
   7. df.groupby().transform() - Group operations
   8. pd.get_dummies()         - One-hot encoding
   9. df.query()               - Filter data
   10. pd.pivot_table()        - Cross-tabulation
   11. df.to_csv()             - Export data

""")

print("="*80)
print("✅ TUTORIAL COMPLETE! Your data is cleaned and ready for analysis!")
print("="*80)

