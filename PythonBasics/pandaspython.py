# ============================================================
# COMPREHENSIVE PANDAS EXAMPLES FOR DATA CLEANING & ML TRAINING
# ============================================================
# This file demonstrates top pandas features used in data cleaning
# and preparing data for machine learning models
# ============================================================




import pandas as pd
import numpy as np

print("="*70)
print("PANDAS DATA CLEANING & ML PREPARATION EXAMPLES")
print("="*70)

# ============================================================
# PART 1: LOADING DATA
# ============================================================

print("\n" + "="*70)
print("PART 1: LOADING DATA WITH PANDAS")
print("="*70)

# Syntax: pd.read_csv(filepath, parse_dates=[], dtype={}, ...)
# Explanation:
# - parse_dates: Automatically converts date strings to datetime objects
# - dtype: Specifies data types for columns (prevents type inference errors)
# - Returns: DataFrame object (2D table with rows and columns)

orders = pd.read_csv(
    '/Users/satya/PythonLearn/PythonBasics/orders.csv',
    parse_dates=['OrderDate']  # Convert OrderDate column to datetime type
)

print("\n1. BASIC DATA INSPECTION:")
print(f"Shape: {orders.shape}")  # (rows, columns)
print(f"\nFirst 5 rows:")
print(orders.head())  # .head() shows first 5 rows by default

print(f"\nData types:")
print(orders.dtypes)  # Shows data type of each column

print(f"\nData info:")
print(orders.info())  # Shows summary: columns, non-null counts, dtypes, memory

print(f"\nBasic statistics:")
print(orders.describe())  # Statistical summary for numeric columns

# ============================================================
# PART 2: DATA CLEANING - HANDLING MISSING VALUES
# ============================================================

print("\n" + "="*70)
print("PART 2: HANDLING MISSING VALUES")
print("="*70)

# Syntax: df.isna() or df.isnull()
# Explanation: Returns DataFrame of True/False showing where values are missing
print("\n1. CHECKING FOR MISSING VALUES:")
print(f"Missing values per column:\n{orders.isna().sum()}")

# Syntax: df.fillna(value)
# Explanation: Fills missing values with specified value
# Can use: df.fillna(0), df.fillna(df['column'].mean()), df.fillna('Unknown')

# Example: Fill missing quantities with 0
if orders['Quantity'].isna().any():
    orders['Quantity'] = orders['Quantity'].fillna(0)
    print("\nFilled missing Quantity values with 0")

# Syntax: df.dropna()
# Explanation: Removes rows with missing values
# Options: df.dropna(subset=['column']), df.dropna(how='all')
# We'll skip this since our data appears clean

print("\n2. DATA TYPES CLEANUP:")
# Convert string columns to proper string type
# Syntax: df['column'].astype('string')
# Explanation: Converts column to string type (more efficient than object)
for col in ['CustomerName', 'Product', 'Category', 'Country']:
    orders[col] = orders[col].astype('string')

# ============================================================
# PART 3: DATA CLEANING - STRING OPERATIONS
# ============================================================

print("\n" + "="*70)
print("PART 3: STRING OPERATIONS FOR DATA CLEANING")
print("="*70)

# Syntax: df['column'].str.method()
# Explanation: .str accessor allows string operations on entire column
# Similar to Python string methods but works on Series

print("\n1. STRING STRIP (Remove whitespace):")
# Syntax: .str.strip()
# Explanation: Removes leading/trailing whitespace from each string
orders['CustomerName'] = orders['CustomerName'].str.strip()
orders['Product'] = orders['Product'].str.strip()
orders['Category'] = orders['Category'].str.strip()
orders['Country'] = orders['Country'].str.strip()
print("Removed whitespace from string columns")

print("\n2. STRING REPLACE (Standardize values):")
# Syntax: df['column'].replace(old_value, new_value)
# Explanation: Replaces specific values in a column
# Can use dict: df['column'].replace({'old1': 'new1', 'old2': 'new2'})
orders['Country'] = orders['Country'].replace('Pakistan', 'Pakistan')  # Example
print("Standardized country names")

print("\n3. STRING CASE CONVERSION:")
# Syntax: .str.lower(), .str.upper(), .str.title()
# Explanation: Changes case of strings
orders['Shipped'] = orders['Shipped'].str.lower()  # Convert Yes/No to yes/no
print("Converted Shipped column to lowercase")

# Convert boolean-like strings to actual boolean
# Syntax: df['column'].map(dictionary)
# Explanation: Maps each value using dictionary (like Python dict.get())
shipped_map = {'yes': True, 'no': False}
orders['Shipped'] = orders['Shipped'].map(shipped_map)
print("Converted Shipped to boolean (True/False)")

# ============================================================
# PART 4: DATA CLEANING - REMOVING DUPLICATES
# ============================================================

print("\n" + "="*70)
print("PART 4: REMOVING DUPLICATES")
print("="*70)

# Syntax: df.duplicated()
# Explanation: Returns boolean Series indicating duplicate rows
print(f"\nNumber of duplicate rows: {orders.duplicated().sum()}")

# Syntax: df.drop_duplicates()
# Explanation: Removes duplicate rows
# Options:
# - subset=['column']: Check duplicates only in specific columns
# - keep='first' or 'last': Which duplicate to keep
# - inplace=True: Modify DataFrame directly (instead of returning new one)
orders = orders.drop_duplicates()
print(f"After removing duplicates: {orders.shape}")

# ============================================================
# PART 5: FEATURE ENGINEERING FOR ML
# ============================================================

print("\n" + "="*70)
print("PART 5: FEATURE ENGINEERING FOR MACHINE LEARNING")
print("="*70)

print("\n1. CREATING NEW COLUMNS (Derived features):")
# Syntax: df['new_column'] = expression
# Explanation: Creates new column based on existing columns
orders['TotalAmount'] = orders['Quantity'] * orders['Price']
print("Created TotalAmount = Quantity * Price")
print(f"Sample TotalAmount values:\n{orders[['Quantity', 'Price', 'TotalAmount']].head()}")

print("\n2. DATETIME OPERATIONS (Extracting date features):")
# Syntax: df['date_column'].dt.attribute
# Explanation: .dt accessor allows datetime operations
# Common attributes: .year, .month, .day, .dayofweek, .quarter, .weekday
orders['OrderYear'] = orders['OrderDate'].dt.year
orders['OrderMonth'] = orders['OrderDate'].dt.month
orders['OrderDay'] = orders['OrderDate'].dt.day
orders['OrderDayOfWeek'] = orders['OrderDate'].dt.dayofweek  # 0=Monday, 6=Sunday
orders['OrderQuarter'] = orders['OrderDate'].dt.quarter
print("Extracted date features: Year, Month, Day, DayOfWeek, Quarter")
print(f"Sample date features:\n{orders[['OrderDate', 'OrderYear', 'OrderMonth', 'OrderDayOfWeek']].head()}")

print("\n3. GROUPBY OPERATIONS (Aggregations):")
# Syntax: df.groupby('column').aggregation()
# Explanation: Groups rows by column values and applies aggregation function
# Common aggregations: .sum(), .mean(), .count(), .max(), .min(), .std()

# Count orders per customer
# Syntax: .transform('function') - applies function but keeps original shape
# Explanation: Returns Series with same length as original DataFrame
customer_order_counts = orders.groupby('CustomerName')['OrderID'].transform('count')
orders['CustomerOrderCount'] = customer_order_counts
print("Created CustomerOrderCount (total orders per customer)")
print(f"Sample customer order counts:\n{orders[['CustomerName', 'CustomerOrderCount']].head(10)}")

# Calculate average order value per customer
customer_avg_order = orders.groupby('CustomerName')['TotalAmount'].transform('mean')
orders['CustomerAvgOrderValue'] = customer_avg_order
print("Created CustomerAvgOrderValue (average order value per customer)")

print("\n4. CATEGORICAL ENCODING (For ML models):")
# One-Hot Encoding: Creates binary columns for each category
# Syntax: pd.get_dummies(df['column'], prefix='prefix_name')
# Explanation: Converts categorical column into multiple binary columns
# Example: Category 'Electronics' becomes Category_Electronics = 1, others = 0

category_dummies = pd.get_dummies(orders['Category'], prefix='Category')
print(f"One-hot encoded Category:\n{category_dummies.head()}")

country_dummies = pd.get_dummies(orders['Country'], prefix='Country')
print(f"\nOne-hot encoded Country (first few columns):\n{country_dummies.iloc[:, :5].head()}")

# Combine original data with encoded columns
# Syntax: pd.concat([df1, df2], axis=1)
# Explanation: Concatenates DataFrames horizontally (axis=1) or vertically (axis=0)
orders_ml = pd.concat([
    orders.drop(columns=['Category', 'Country']),  # Drop original categorical columns
    category_dummies,
    country_dummies
], axis=1)

print(f"\nCombined DataFrame shape: {orders_ml.shape}")
print(f"Original columns: {orders.shape[1]}, ML-ready columns: {orders_ml.shape[1]}")

# ============================================================
# PART 6: DATA FILTERING AND QUERYING
# ============================================================

print("\n" + "="*70)
print("PART 6: DATA FILTERING AND QUERYING")
print("="*70)

print("\n1. BOOLEAN INDEXING:")
# Syntax: df[df['column'] condition]
# Explanation: Filters rows where condition is True
shipped_orders = orders[orders['Shipped'] == True]
print(f"Shipped orders: {len(shipped_orders)} out of {len(orders)}")

high_value_orders = orders[orders['TotalAmount'] > 200]
print(f"High value orders (>$200): {len(high_value_orders)}")

# Multiple conditions use & (and), | (or), ~ (not)
# Note: Must use parentheses around each condition
electronics_orders = orders[(orders['Category'] == 'Electronics') & (orders['TotalAmount'] > 100)]
print(f"Electronics orders >$100: {len(electronics_orders)}")

print("\n2. QUERY METHOD (More readable filtering):")
# Syntax: df.query('condition')
# Explanation: Allows SQL-like query syntax (more readable for complex conditions)
# Note: Use == for equality, != for not equal, and/or for logical operators
clean_subset = orders.query('Shipped == True and TotalAmount > 0')
print(f"Clean shipped orders: {len(clean_subset)}")

print("\n3. SELECTING COLUMNS:")
# Syntax: df[['col1', 'col2']] or df.loc[:, ['col1', 'col2']]
# Explanation: Selects specific columns
selected = orders[['OrderID', 'CustomerName', 'TotalAmount', 'OrderDate']]
print(f"Selected columns:\n{selected.head()}")

# Select columns by pattern (for ML features)
feature_columns = [col for col in orders_ml.columns 
                   if col.startswith('Category_') or col.startswith('Country_')]
print(f"\nFeature columns (one-hot encoded): {len(feature_columns)}")

# ============================================================
# PART 7: AGGREGATIONS AND PIVOT TABLES
# ============================================================

print("\n" + "="*70)
print("PART 7: AGGREGATIONS AND PIVOT TABLES")
print("="*70)

print("\n1. GROUPBY WITH AGGREGATIONS:")
# Syntax: df.groupby('column').agg({'column': ['function1', 'function2']})
# Explanation: Groups data and applies multiple aggregation functions
category_stats = orders.groupby('Category').agg({
    'TotalAmount': ['count', 'sum', 'mean', 'max'],
    'Quantity': 'sum'
}).reset_index()

# Flatten column names
category_stats.columns = ['Category', 'OrderCount', 'TotalRevenue', 'AvgOrderValue', 'MaxOrderValue', 'TotalQuantity']
print("Category Statistics:")
print(category_stats)

print("\n2. PIVOT TABLES:")
# Syntax: pd.pivot_table(df, index='row', columns='col', values='value', aggfunc='function')
# Explanation: Creates cross-tabulation (like Excel pivot table)
# - index: Rows
# - columns: Columns
# - values: Values to aggregate
# - aggfunc: Aggregation function (sum, mean, count, etc.)
country_month_revenue = pd.pivot_table(
    orders,
    index='Country',
    columns='OrderMonth',
    values='TotalAmount',
    aggfunc='sum',
    fill_value=0  # Fill missing combinations with 0
)
print("\nRevenue by Country and Month:")
print(country_month_revenue.head(10))

print("\n3. CROSS-TABULATION:")
# Syntax: pd.crosstab(index, columns)
# Explanation: Frequency table for categorical variables
shipped_by_category = pd.crosstab(orders['Category'], orders['Shipped'])
print("Shipped status by Category:")
print(shipped_by_category)

# ============================================================
# PART 8: PREPARING DATA FOR ML MODELS
# ============================================================

print("\n" + "="*70)
print("PART 8: PREPARING DATA FOR MACHINE LEARNING")
print("="*70)

print("\n1. SELECTING FEATURES (X) AND TARGET (y):")
# For ML, we need:
# - X: Feature matrix (all input variables)
# - y: Target variable (what we want to predict)

# Select numeric features
numeric_features = ['Quantity', 'Price', 'TotalAmount', 'OrderYear', 
                    'OrderMonth', 'OrderDayOfWeek', 'CustomerOrderCount', 'CustomerAvgOrderValue']

# Select one-hot encoded features
categorical_features = [col for col in orders_ml.columns 
                       if col.startswith('Category_') or col.startswith('Country_')]

# Combine all features
X = orders_ml[numeric_features + categorical_features]
print(f"Feature matrix (X) shape: {X.shape}")
print(f"Features: {len(numeric_features)} numeric + {len(categorical_features)} categorical")

# Example target: predict if order will be shipped (classification)
y = orders_ml['Shipped'].astype(int)  # Convert True/False to 1/0
print(f"Target (y) shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

print("\n2. HANDLING OUTLIERS (Optional):")
# Syntax: df[(df['column'] < upper_bound) & (df['column'] > lower_bound)]
# Explanation: Remove extreme values that might skew ML models
Q1 = orders['TotalAmount'].quantile(0.25)
Q3 = orders['TotalAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"TotalAmount statistics:")
print(f"  Q1 (25th percentile): ${Q1:.2f}")
print(f"  Q3 (75th percentile): ${Q3:.2f}")
print(f"  IQR: ${IQR:.2f}")
print(f"  Outlier bounds: ${lower_bound:.2f} to ${upper_bound:.2f}")

# Filter outliers (optional - comment out if you want to keep all data)
outliers = orders[(orders['TotalAmount'] < lower_bound) | (orders['TotalAmount'] > upper_bound)]
print(f"  Potential outliers: {len(outliers)} orders")

print("\n3. SPLITTING DATA (Train/Test split):")
# In real ML, you'd use: from sklearn.model_selection import train_test_split
# For demonstration, we'll show the concept
train_size = int(0.8 * len(orders_ml))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train/Test split: 80/20")

print("\n4. FEATURE SCALING PREPARATION:")
# ML models often require scaled features
# Syntax: StandardScaler or MinMaxScaler from sklearn
# For now, we'll just show the data is ready
print("Features are ready for scaling:")
print(f"  Numeric features range: {X[numeric_features].min().min():.2f} to {X[numeric_features].max().max():.2f}")

# ============================================================
# PART 9: DATA EXPORT FOR ML PIPELINE
# ============================================================

print("\n" + "="*70)
print("PART 9: SAVING CLEANED DATA")
print("="*70)

# Syntax: df.to_csv(), df.to_parquet(), df.to_excel()
# Explanation: Saves DataFrame to file

# Save as CSV (human-readable)
csv_path = '/Users/satya/PythonLearn/PythonBasics/orders_cleaned.csv'
orders.to_csv(csv_path, index=False)  # index=False prevents saving row indices
print(f"Saved cleaned data to CSV: {csv_path}")

# Save as Parquet (more efficient, preserves data types)
try:
    parquet_path = '/Users/satya/PythonLearn/PythonBasics/orders_cleaned.parquet'
    orders.to_parquet(parquet_path, index=False)
    print(f"Saved cleaned data to Parquet: {parquet_path}")
except Exception as e:
    print(f"Parquet save skipped (may need pyarrow): {e}")

# Save ML-ready features
ml_path = '/Users/satya/PythonLearn/PythonBasics/orders_ml_ready.csv'
X.to_csv(ml_path, index=False)
print(f"Saved ML-ready features to: {ml_path}")

# ============================================================
# PART 10: SUMMARY AND QUICK REFERENCE
# ============================================================

print("\n" + "="*70)
print("PART 10: QUICK REFERENCE - TOP PANDAS FEATURES")
print("="*70)


print("""
📚 TOP PANDAS FEATURES FOR DATA CLEANING & ML:

1. DATA LOADING:
   ✅ pd.read_csv(filepath, parse_dates=[], dtype={})
      - Loads CSV with type specifications
      - Automatically parses dates

2. DATA INSPECTION:
   ✅ df.head(), df.tail(), df.info(), df.describe()
   ✅ df.shape, df.dtypes, df.columns

3. MISSING VALUES:
   ✅ df.isna(), df.isnull() - Check for missing
   ✅ df.fillna(value) - Fill missing values
   ✅ df.dropna() - Remove rows with missing

4. STRING OPERATIONS:
   ✅ df['col'].str.strip() - Remove whitespace
   ✅ df['col'].str.lower() - Convert to lowercase
   ✅ df['col'].replace(old, new) - Replace values
   ✅ df['col'].map(dict) - Map values using dictionary

5. DUPLICATES:
   ✅ df.duplicated() - Find duplicates
   ✅ df.drop_duplicates() - Remove duplicates

6. FEATURE ENGINEERING:
   ✅ df['new_col'] = expression - Create new columns
   ✅ df['date'].dt.year/.month/.day - Extract date parts
   ✅ df.groupby().transform() - Group operations keeping shape

7. CATEGORICAL ENCODING:
   ✅ pd.get_dummies(df['col']) - One-hot encoding
   ✅ pd.concat([df1, df2], axis=1) - Combine DataFrames

8. FILTERING:
   ✅ df[df['col'] > value] - Boolean indexing
   ✅ df.query('condition') - SQL-like queries

9. AGGREGATIONS:
   ✅ df.groupby().agg() - Group with multiple functions
   ✅ pd.pivot_table() - Cross-tabulation
   ✅ pd.crosstab() - Frequency tables

10. DATA EXPORT:
   ✅ df.to_csv() - Save as CSV
   ✅ df.to_parquet() - Save as Parquet (efficient)

📊 FINAL DATA SUMMARY:
""")

print(f"Original data shape: {orders.shape}")
print(f"Cleaned data shape: {orders.shape}")
print(f"ML-ready features shape: {X.shape}")
print(f"Number of features: {len(numeric_features + categorical_features)}")
print(f"  - Numeric: {len(numeric_features)}")
print(f"  - Categorical (one-hot): {len(categorical_features)}")
print(f"Target variable: Shipped (binary classification)")

print("\n" + "="*70)
print("✅ PANDAS DATA CLEANING & ML PREPARATION COMPLETE!")
print("="*70)