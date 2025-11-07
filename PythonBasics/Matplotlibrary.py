# ==================================================================================
# COMPREHENSIVE MATPLOTLIB TUTORIAL - DATA & FEATURE ENGINEERING VISUALIZATION
# ==================================================================================
# 
# PURPOSE: This script demonstrates how to use matplotlib for visualizing data
# during data engineering and feature engineering processes.
#
# USE CASE: Visualizing e-commerce order data to understand patterns, detect
# issues, and guide feature engineering decisions.
#
# TARGET AUDIENCE: Beginners to matplotlib and data visualization
#
# WHAT YOU'LL LEARN:
#   1. Basic plotting (line, bar, scatter, histogram)
#   2. Subplots and figure layout
#   3. Customizing plots (colors, labels, titles)
#   4. Data exploration visualizations
#   5. Feature engineering visualizations
#   6. Detecting outliers and anomalies
#   7. Time series analysis
#   8. Distribution analysis
#
# ==================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Set style for better-looking plots
# Try modern style, fallback to default if not available
try:
    plt.style.use('seaborn-v0_8-darkgrid')  # Modern, clean style
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')  # Alternative seaborn style
    except OSError:
        plt.style.use('default')  # Default matplotlib style

# Set figure size and DPI for better quality
plt.rcParams['figure.figsize'] = (12, 6)  # Default figure size (width, height)
plt.rcParams['figure.dpi'] = 100  # Resolution (dots per inch)

print("="*80)
print("MATPLOTLIB DATA & FEATURE ENGINEERING VISUALIZATION TUTORIAL")
print("="*80)
print("\n📊 Use Case: E-commerce Order Data Visualization")
print("📁 Dataset: orders.csv")
print("🎯 Goal: Visualize data to guide data cleaning and feature engineering")
print("\n" + "="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING AND PREPARING DATA")
print("="*80)

# Load the orders data
orders = pd.read_csv(
    '/Users/satya/PythonLearn/PythonBasics/orders.csv',
    parse_dates=['OrderDate']
)

# Basic data cleaning (same as pandas tutorial)
orders['TotalAmount'] = orders['Quantity'] * orders['Price']
orders['OrderYear'] = orders['OrderDate'].dt.year
orders['OrderMonth'] = orders['OrderDate'].dt.month
orders['OrderDayOfWeek'] = orders['OrderDate'].dt.dayofweek
orders['OrderWeekday'] = orders['OrderDate'].dt.day_name()
orders['Shipped'] = orders['Shipped'].str.lower().map({'yes': True, 'no': False})

print(f"✅ Data loaded: {orders.shape[0]} orders")
print(f"✅ Features created: TotalAmount, OrderYear, OrderMonth, OrderDayOfWeek")

# ============================================================================
# STEP 2: BASIC PLOTTING - Understanding matplotlib Fundamentals
# ============================================================================

print("\n" + "="*80)
print("STEP 2: BASIC PLOTTING - Understanding matplotlib Fundamentals")
print("="*80)

# WHAT IS MATPLOTLIB?
# - Python's primary plotting library
# - Creates static, animated, and interactive visualizations
# - Think of it as Python's version of Excel charts

# BASIC CONCEPT:
# - plt.figure() - Creates a new figure (canvas)
# - plt.plot(), plt.bar(), plt.scatter() - Draws on the figure
# - plt.xlabel(), plt.ylabel(), plt.title() - Adds labels
# - plt.show() - Displays the plot

print("\n1️⃣ CREATING YOUR FIRST PLOT - Line Chart:")
print("-" * 80)

# Line Chart - Shows trends over time
# WHY: Perfect for time series data (revenue over time, orders per day, etc.)

# Create a figure
plt.figure(figsize=(10, 5))

# Calculate daily revenue
daily_revenue = orders.groupby('OrderDate')['TotalAmount'].sum().reset_index()

# Plot the data
# SYNTAX: plt.plot(x_values, y_values, style_options)
# - x: x-axis values (dates)
# - y: y-axis values (revenue)
# - 'b-' means blue line, '-' means solid line
plt.plot(daily_revenue['OrderDate'], daily_revenue['TotalAmount'], 
         'b-', linewidth=2, label='Daily Revenue')

# Customize the plot
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Revenue ($)', fontsize=12, fontweight='bold')
plt.title('Daily Revenue Trend - E-commerce Orders', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)  # Add grid (alpha = transparency)
plt.legend()  # Show legend
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# Format dates on x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('/Users/satya/PythonLearn/PythonBasics/daily_revenue_trend.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: daily_revenue_trend.png")
print("   📊 Shows: Revenue trend over time (helps identify patterns)")

# Display the plot
plt.show()

# ============================================================================
# STEP 3: BAR CHARTS - Comparing Categories
# ============================================================================

print("\n" + "="*80)
print("STEP 3: BAR CHARTS - Comparing Categories")
print("="*80)

# Bar Chart - Compares values across categories
# WHY: Great for comparing revenue by category, orders by country, etc.

print("\n1️⃣ REVENUE BY PRODUCT CATEGORY:")
print("-" * 80)

# Calculate revenue by category
category_revenue = orders.groupby('Category')['TotalAmount'].sum().sort_values(ascending=False)

# Create figure
plt.figure(figsize=(10, 6))

# Create bar chart
# SYNTAX: plt.bar(x_positions, heights, color, width)
bars = plt.bar(category_revenue.index, category_revenue.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Custom colors
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}',  # Format as currency
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Product Category', fontsize=12, fontweight='bold')
plt.ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
plt.title('Total Revenue by Product Category', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)  # Only horizontal grid lines
plt.tight_layout()

plt.savefig('/Users/satya/PythonLearn/PythonBasics/revenue_by_category.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: revenue_by_category.png")
print("   📊 Shows: Which categories generate most revenue (feature importance)")

plt.show()

# ============================================================================
# STEP 4: HISTOGRAMS - Understanding Distributions
# ============================================================================

print("\n" + "="*80)
print("STEP 4: HISTOGRAMS - Understanding Data Distributions")
print("="*80)

# Histogram - Shows distribution of numeric values
# WHY: Essential for feature engineering - understand if data is normal, skewed, etc.
#      Helps detect outliers and decide on feature transformations

print("\n1️⃣ DISTRIBUTION OF ORDER AMOUNTS:")
print("-" * 80)

# Create figure with subplots (multiple plots in one figure)
# SYNTAX: fig, axes = plt.subplots(rows, cols, figsize=(width, height))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Histogram
# SYNTAX: plt.hist(data, bins, color, edgecolor)
axes[0].hist(orders['TotalAmount'], bins=20, color='skyblue', 
            edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Order Amount ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency (Number of Orders)', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Order Amounts', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axvline(orders['TotalAmount'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: ${orders["TotalAmount"].mean():.2f}')
axes[0].axvline(orders['TotalAmount'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: ${orders["TotalAmount"].median():.2f}')
axes[0].legend()

# Right plot: Box plot (shows quartiles and outliers)
# WHY: Better for detecting outliers than histogram
axes[1].boxplot(orders['TotalAmount'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Order Amount ($)', fontsize=11, fontweight='bold')
axes[1].set_title('Box Plot - Order Amounts (Outlier Detection)', 
                 fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/order_amount_distribution.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: order_amount_distribution.png")
print("   📊 Shows: Distribution shape (normal? skewed?) and outliers")
print("   💡 Feature Engineering Insight: May need log transformation if skewed")

plt.show()

# ============================================================================
# STEP 5: SCATTER PLOTS - Finding Relationships
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SCATTER PLOTS - Finding Relationships Between Features")
print("="*80)

# Scatter Plot - Shows relationship between two numeric variables
# WHY: Critical for feature engineering - identify correlations, patterns
#      Helps decide which features to combine or remove

print("\n1️⃣ RELATIONSHIP BETWEEN PRICE AND QUANTITY:")
print("-" * 80)

plt.figure(figsize=(10, 6))

# Create scatter plot
# SYNTAX: plt.scatter(x, y, s=size, c=color, alpha=transparency)
scatter = plt.scatter(orders['Price'], orders['Quantity'], 
                     s=orders['TotalAmount']/10,  # Size based on total amount
                     c=orders['TotalAmount'],      # Color based on total amount
                     cmap='viridis',              # Color map
                     alpha=0.6, edgecolors='black', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Total Amount ($)', fontsize=11, fontweight='bold')

plt.xlabel('Price per Unit ($)', fontsize=12, fontweight='bold')
plt.ylabel('Quantity', fontsize=12, fontweight='bold')
plt.title('Price vs Quantity Relationship\n(Bubble size = Total Amount)', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/price_quantity_relationship.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: price_quantity_relationship.png")
print("   📊 Shows: Relationship between price and quantity")
print("   💡 Feature Engineering Insight: TotalAmount = Price × Quantity is useful!")

plt.show()

# ============================================================================
# STEP 6: MULTIPLE SUBPLOTS - Comprehensive Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 6: MULTIPLE SUBPLOTS - Comprehensive Feature Analysis")
print("="*80)

# Subplots - Multiple plots in one figure
# WHY: Compare multiple aspects of data side-by-side
#      Essential for data exploration and feature engineering

print("\n1️⃣ TIME-BASED FEATURE ANALYSIS:")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Time-Based Feature Engineering Analysis', 
             fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Orders by Month
monthly_orders = orders.groupby('OrderMonth')['OrderID'].count()
axes[0, 0].bar(monthly_orders.index, monthly_orders.values, 
               color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Month', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Number of Orders', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Orders by Month', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(range(6, 8))  # June and July
axes[0, 0].set_xticklabels(['June', 'July'])
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Revenue by Day of Week
dow_revenue = orders.groupby('OrderDayOfWeek')['TotalAmount'].sum()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0, 1].bar(range(len(dow_revenue)), dow_revenue.values, 
               color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Day of Week', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Total Revenue ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Revenue by Day of Week', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(day_names)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Average Order Value by Category
category_avg = orders.groupby('Category')['TotalAmount'].mean().sort_values(ascending=False)
axes[1, 0].barh(category_avg.index, category_avg.values, 
                color='mediumseagreen', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Average Order Value ($)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Category', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Average Order Value by Category', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Shipped vs Not Shipped
shipped_counts = orders['Shipped'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
axes[1, 1].pie(shipped_counts.values, labels=['Shipped', 'Not Shipped'], 
               autopct='%1.1f%%', colors=colors, startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1, 1].set_title('Order Shipping Status', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/time_based_analysis.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: time_based_analysis.png")
print("   📊 Shows: Multiple time-based patterns for feature engineering")

plt.show()

# ============================================================================
# STEP 7: FEATURE ENGINEERING VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 7: FEATURE ENGINEERING VISUALIZATION")
print("="*80)

# Visualizing engineered features to validate their usefulness
# WHY: Before using features in ML, visualize to ensure they're meaningful

print("\n1️⃣ CUSTOMER SEGMENTATION FEATURES:")
print("-" * 80)

# Create customer features
customer_features = orders.groupby('CustomerName').agg({
    'OrderID': 'count',
    'TotalAmount': ['sum', 'mean']
}).reset_index()
customer_features.columns = ['CustomerName', 'OrderCount', 'TotalSpent', 'AvgOrderValue']

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Customer Order Count Distribution
axes[0].hist(customer_features['OrderCount'], bins=15, color='purple', 
            alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Number of Orders per Customer', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
axes[0].set_title('Customer Order Count Distribution', fontsize=12, fontweight='bold')
axes[0].axvline(customer_features['OrderCount'].mean(), color='red', 
                linestyle='--', linewidth=2, 
                label=f'Mean: {customer_features["OrderCount"].mean():.1f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Total Spent vs Order Count (Scatter)
scatter = axes[1].scatter(customer_features['OrderCount'], 
                         customer_features['TotalSpent'],
                         s=100, c=customer_features['AvgOrderValue'], 
                         cmap='coolwarm', alpha=0.7, edgecolors='black')
axes[1].set_xlabel('Number of Orders', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total Spent ($)', fontsize=11, fontweight='bold')
axes[1].set_title('Customer Value Analysis\n(Color = Avg Order Value)', 
                 fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('Avg Order Value ($)', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/customer_features.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: customer_features.png")
print("   📊 Shows: Engineered customer features (useful for ML segmentation)")

plt.show()

# ============================================================================
# STEP 8: OUTLIER DETECTION VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 8: OUTLIER DETECTION VISUALIZATION")
print("="*80)

# Visualizing outliers helps decide how to handle them
# WHY: Outliers can skew ML models - need to identify and handle them

print("\n1️⃣ IDENTIFYING OUTLIERS IN ORDER AMOUNTS:")
print("-" * 80)

# Calculate IQR (Interquartile Range) for outlier detection
Q1 = orders['TotalAmount'].quantile(0.25)
Q3 = orders['TotalAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = orders[(orders['TotalAmount'] < lower_bound) | 
                 (orders['TotalAmount'] > upper_bound)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Box plot with outliers highlighted
bp = axes[0].boxplot(orders['TotalAmount'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', 
                                  markersize=8, alpha=0.7))
axes[0].axhline(upper_bound, color='orange', linestyle='--', linewidth=2, 
               label=f'Upper Bound: ${upper_bound:.2f}')
axes[0].axhline(lower_bound, color='orange', linestyle='--', linewidth=2, 
               label=f'Lower Bound: ${lower_bound:.2f}')
axes[0].set_ylabel('Order Amount ($)', fontsize=11, fontweight='bold')
axes[0].set_title('Box Plot - Outlier Detection', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Scatter plot highlighting outliers
normal_orders = orders[(orders['TotalAmount'] >= lower_bound) & 
                      (orders['TotalAmount'] <= upper_bound)]
axes[1].scatter(normal_orders.index, normal_orders['TotalAmount'], 
               color='blue', alpha=0.6, label='Normal Orders', s=50)
axes[1].scatter(outliers.index, outliers['TotalAmount'], 
               color='red', alpha=0.8, label='Outliers', s=100, 
               marker='^', edgecolors='black')
axes[1].axhline(upper_bound, color='orange', linestyle='--', linewidth=2)
axes[1].axhline(lower_bound, color='orange', linestyle='--', linewidth=2)
axes[1].set_xlabel('Order Index', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Order Amount ($)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Outliers Visualization ({len(outliers)} outliers found)', 
                 fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/outlier_detection.png', 
            dpi=150, bbox_inches='tight')
print(f"   ✅ Created: outlier_detection.png")
print(f"   📊 Shows: {len(outliers)} outliers identified")
print("   💡 Feature Engineering: May need to cap or remove outliers")

plt.show()

# ============================================================================
# STEP 9: CORRELATION HEATMAP - Feature Relationships
# ============================================================================

print("\n" + "="*80)
print("STEP 9: CORRELATION HEATMAP - Understanding Feature Relationships")
print("="*80)

# Correlation Heatmap - Shows how features relate to each other
# WHY: Critical for feature engineering - identify redundant features
#      Find features that should be combined or removed

print("\n1️⃣ FEATURE CORRELATION ANALYSIS:")
print("-" * 80)

# Select numeric features for correlation
numeric_features = orders.select_dtypes(include=[np.number])
correlation_matrix = numeric_features.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Use imshow or pcolor for heatmap
im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix.columns)

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

ax.set_title('Feature Correlation Heatmap\n(For Feature Engineering)', 
            fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/correlation_heatmap.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: correlation_heatmap.png")
print("   📊 Shows: Which features are highly correlated (may be redundant)")
print("   💡 Feature Engineering: Remove highly correlated features to avoid multicollinearity")

plt.show()

# ============================================================================
# STEP 10: TIME SERIES ANALYSIS - Trend Visualization
# ============================================================================

print("\n" + "="*80)
print("STEP 10: TIME SERIES ANALYSIS - Trend Visualization")
print("="*80)

# Time Series - Shows how values change over time
# WHY: Essential for creating time-based features (moving averages, trends, etc.)

print("\n1️⃣ REVENUE TREND WITH MOVING AVERAGE:")
print("-" * 80)

# Prepare time series data
daily_data = orders.groupby('OrderDate').agg({
    'TotalAmount': 'sum',
    'OrderID': 'count'
}).reset_index()
daily_data.columns = ['Date', 'Revenue', 'OrderCount']

# Calculate 7-day moving average (smooths out daily fluctuations)
daily_data['Revenue_MA7'] = daily_data['Revenue'].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(14, 6))

# Plot actual daily revenue
plt.plot(daily_data['Date'], daily_data['Revenue'], 
         'o-', color='steelblue', linewidth=1.5, markersize=6, 
         alpha=0.7, label='Daily Revenue')

# Plot moving average
plt.plot(daily_data['Date'], daily_data['Revenue_MA7'], 
         '-', color='red', linewidth=2.5, label='7-Day Moving Average')

plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Revenue ($)', fontsize=12, fontweight='bold')
plt.title('Revenue Trend with Moving Average\n(Feature Engineering: Trend Feature)', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Format dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/revenue_trend_ma.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: revenue_trend_ma.png")
print("   📊 Shows: Revenue trend with moving average")
print("   💡 Feature Engineering: Moving average is a useful time-based feature")

plt.show()

# ============================================================================
# STEP 11: ADVANCED VISUALIZATION - Multi-dimensional Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 11: ADVANCED VISUALIZATION - Multi-dimensional Analysis")
print("="*80)

# Combining multiple dimensions in one plot
# WHY: Understand complex relationships between multiple features

print("\n1️⃣ CATEGORY PERFORMANCE BY COUNTRY:")
print("-" * 80)

# Create pivot table: Revenue by Category and Country
pivot_data = orders.pivot_table(
    values='TotalAmount',
    index='Category',
    columns='Country',
    aggfunc='sum',
    fill_value=0
)

# Select top countries by revenue
top_countries = orders.groupby('Country')['TotalAmount'].sum().nlargest(5).index
pivot_data = pivot_data[top_countries]

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')

# Set ticks
ax.set_xticks(range(len(pivot_data.columns)))
ax.set_yticks(range(len(pivot_data.index)))
ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
ax.set_yticklabels(pivot_data.index)

# Add text annotations
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        value = pivot_data.iloc[i, j]
        if value > 0:
            text = ax.text(j, i, f'${value:,.0f}',
                          ha="center", va="center", 
                          color="black" if value < pivot_data.values.max()/2 else "white",
                          fontsize=9, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Revenue ($)', fontsize=11, fontweight='bold')

ax.set_title('Revenue Heatmap: Category × Country\n(Feature Engineering: Interaction Features)', 
            fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/category_country_heatmap.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: category_country_heatmap.png")
print("   📊 Shows: Category performance across countries")
print("   💡 Feature Engineering: Category × Country interaction could be useful")

plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL SUMMARY - MATPLOTLIB VISUALIZATION COMPLETE")
print("="*80)

print(f"""
✅ VISUALIZATIONS CREATED:

1. 📈 Daily Revenue Trend (Line Chart)
   - File: daily_revenue_trend.png
   - Purpose: Identify revenue patterns over time

2. 📊 Revenue by Category (Bar Chart)
   - File: revenue_by_category.png
   - Purpose: Compare category performance

3. 📉 Order Amount Distribution (Histogram + Box Plot)
   - File: order_amount_distribution.png
   - Purpose: Understand data distribution and detect outliers

4. 🔍 Price vs Quantity Relationship (Scatter Plot)
   - File: price_quantity_relationship.png
   - Purpose: Find correlations between features

5. 📅 Time-Based Analysis (Multiple Subplots)
   - File: time_based_analysis.png
   - Purpose: Comprehensive time feature analysis

6. 👥 Customer Features (Feature Engineering)
   - File: customer_features.png
   - Purpose: Visualize engineered customer features

7. ⚠️  Outlier Detection
   - File: outlier_detection.png
   - Purpose: Identify data quality issues

8. 🔥 Correlation Heatmap
   - File: correlation_heatmap.png
   - Purpose: Find redundant features

9. 📈 Revenue Trend with Moving Average
   - File: revenue_trend_ma.png
   - Purpose: Time series feature engineering

10. 🌍 Category × Country Heatmap
    - File: category_country_heatmap.png
    - Purpose: Multi-dimensional feature analysis

📚 KEY MATPLOTLIB FUNCTIONS LEARNED:

BASIC PLOTTING:
   ✅ plt.figure() - Create figure
   ✅ plt.plot() - Line chart
   ✅ plt.bar() - Bar chart
   ✅ plt.scatter() - Scatter plot
   ✅ plt.hist() - Histogram
   ✅ plt.boxplot() - Box plot
   ✅ plt.pie() - Pie chart

CUSTOMIZATION:
   ✅ plt.xlabel(), plt.ylabel() - Axis labels
   ✅ plt.title() - Plot title
   ✅ plt.legend() - Show legend
   ✅ plt.grid() - Add grid
   ✅ plt.colorbar() - Color scale
   ✅ plt.tight_layout() - Adjust spacing

ADVANCED:
   ✅ plt.subplots() - Multiple plots
   ✅ plt.savefig() - Save figure
   ✅ plt.show() - Display plot
   ✅ Custom colors, styles, markers

💡 FEATURE ENGINEERING INSIGHTS FROM VISUALIZATIONS:

1. ✅ TotalAmount is a useful feature (Price × Quantity)
2. ✅ Time features (Month, DayOfWeek) show patterns
3. ✅ Customer features (OrderCount, AvgValue) useful for segmentation
4. ✅ Category features show clear differences
5. ✅ Outliers detected - may need handling
6. ✅ Some features may be correlated (check heatmap)
7. ✅ Moving averages useful for time series
8. ✅ Interaction features (Category × Country) could be valuable

🎯 NEXT STEPS FOR ML:
   - Use visualizations to select best features
   - Handle outliers based on box plots
   - Remove highly correlated features
   - Create interaction features where patterns exist
   - Use time-based features for time series models

""")

print("="*80)
print("✅ MATPLOTLIB TUTORIAL COMPLETE!")
print("="*80)
print("\n💡 TIP: Visualizations are crucial for data engineering!")
print("   They help you understand your data before building ML models.")
print("="*80)