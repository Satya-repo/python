# ==================================================================================
# COMPREHENSIVE SEABORN TUTORIAL - STATISTICAL DATA VISUALIZATION
# ==================================================================================
# 
# PURPOSE: This script demonstrates how to use seaborn for statistical data
# visualization during data engineering and feature engineering processes.
#
# USE CASE: Visualizing e-commerce order data with statistical insights to
# understand distributions, relationships, and patterns.
#
# TARGET AUDIENCE: Beginners to seaborn and statistical visualization
#
# WHAT IS SEABORN?
# - Built on top of matplotlib
# - Makes statistical visualizations easier and more beautiful
# - Automatic statistical aggregations and error bars
# - Beautiful default styles and color palettes
# - Works seamlessly with pandas DataFrames
#
# WHAT YOU'LL LEARN:
#   1. Distribution plots (histograms, KDE, distributions)
#   2. Categorical plots (box plots, violin plots, bar plots)
#   3. Relationship plots (scatter, line, multi-panel)
#   4. Regression plots (linear regression, trend lines)
#   5. Heatmaps and correlation matrices
#   6. Pair plots (comparing multiple variables)
#   7. Advanced features (FacetGrid, styling)
#   8. Feature engineering visualizations
#
# ==================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Set seaborn style (beautiful defaults)
sns.set_style("whitegrid")  # Options: "darkgrid", "whitegrid", "dark", "white", "ticks"
sns.set_palette("husl")  # Color palette: "deep", "muted", "bright", "pastel", "dark", "colorblind"

# Set figure parameters
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("="*80)
print("SEABORN STATISTICAL VISUALIZATION TUTORIAL")
print("="*80)
print("\n📊 Use Case: E-commerce Order Data Statistical Analysis")
print("📁 Dataset: orders.csv")
print("🎯 Goal: Statistical visualizations for data & feature engineering")
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

# Data cleaning and feature engineering
orders['TotalAmount'] = orders['Quantity'] * orders['Price']
orders['OrderYear'] = orders['OrderDate'].dt.year
orders['OrderMonth'] = orders['OrderDate'].dt.month
orders['OrderDayOfWeek'] = orders['OrderDate'].dt.dayofweek
orders['OrderWeekday'] = orders['OrderDate'].dt.day_name()
orders['Shipped'] = orders['Shipped'].str.lower().map({'yes': True, 'no': False})

# Create customer features
customer_features = orders.groupby('CustomerName').agg({
    'OrderID': 'count',
    'TotalAmount': ['sum', 'mean']
}).reset_index()
customer_features.columns = ['CustomerName', 'OrderCount', 'TotalSpent', 'AvgOrderValue']
orders = orders.merge(customer_features, on='CustomerName', how='left')

print(f"✅ Data loaded: {orders.shape[0]} orders")
print(f"✅ Features created: TotalAmount, OrderYear, OrderMonth, Customer features")

# ============================================================================
# STEP 2: DISTRIBUTION PLOTS - Understanding Data Distributions
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DISTRIBUTION PLOTS - Understanding Data Distributions")
print("="*80)

# WHY DISTRIBUTION PLOTS?
# - Understand how data is distributed (normal, skewed, etc.)
# - Detect outliers and anomalies
# - Guide feature transformation decisions (log, square root, etc.)

print("\n1️⃣ HISTPLOT - Histogram with KDE (Kernel Density Estimation):")
print("-" * 80)

# sns.histplot() - Creates histogram with optional KDE overlay
# SYNTAX: sns.histplot(data=df, x='column', kde=True, bins=30)
# - data: DataFrame
# - x: Column to plot
# - kde: Add kernel density estimate (smooth curve)
# - bins: Number of bins
# - hue: Color by another column (for comparison)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram with KDE
sns.histplot(data=orders, x='TotalAmount', kde=True, bins=20, 
             color='skyblue', edgecolor='black', ax=axes[0])
axes[0].set_title('Distribution of Order Amounts\n(With KDE)', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Total Amount ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')

# Right: Histogram grouped by category (hue)
sns.histplot(data=orders, x='TotalAmount', hue='Category', 
             kde=True, bins=15, alpha=0.7, ax=axes[1])
axes[1].set_title('Order Amount Distribution by Category', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Total Amount ($)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].legend(title='Category', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_distribution.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_distribution.png")
print("   📊 Shows: Data distribution and category differences")
print("   💡 Feature Engineering: Identify if log transformation needed")

plt.show()

print("\n2️⃣ DISPLOT - Distribution Plot (Alternative):")
print("-" * 80)

# sns.displot() - More flexible distribution plot (can create subplots)
# SYNTAX: sns.displot(data=df, x='column', kind='hist', col='category')
# - kind: 'hist', 'kde', 'ecdf' (empirical cumulative distribution)
# - col: Create separate plots for each category

g = sns.displot(data=orders, x='TotalAmount', hue='Category', 
                kind='kde', fill=True, alpha=0.6, height=5, aspect=1.5)
g.fig.suptitle('KDE Plot - Order Amounts by Category', 
               fontsize=14, fontweight='bold', y=1.02)
g.set_axis_labels('Total Amount ($)', 'Density', fontweight='bold')
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_kde.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_kde.png")
print("   📊 Shows: Smooth distribution curves for each category")

plt.show()

# ============================================================================
# STEP 3: CATEGORICAL PLOTS - Comparing Categories
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CATEGORICAL PLOTS - Comparing Categories")
print("="*80)

# WHY CATEGORICAL PLOTS?
# - Compare values across different categories
# - Understand category differences
# - Identify which categories perform better

print("\n1️⃣ BOXPLOT - Box and Whisker Plot:")
print("-" * 80)

# sns.boxplot() - Shows quartiles, median, and outliers
# SYNTAX: sns.boxplot(data=df, x='category', y='value')
# - Shows: Q1, median, Q3, outliers
# - Great for comparing distributions across categories

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Box plot by category
sns.boxplot(data=orders, x='Category', y='TotalAmount', 
            palette='Set2', ax=axes[0])
axes[0].set_title('Order Amount Distribution by Category\n(Box Plot)', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Total Amount ($)', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Right: Box plot with hue (additional grouping)
sns.boxplot(data=orders, x='Category', y='TotalAmount', 
            hue='Shipped', palette='pastel', ax=axes[1])
axes[1].set_title('Order Amount by Category and Shipping Status', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total Amount ($)', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Shipped', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_boxplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_boxplot.png")
print("   📊 Shows: Distribution comparison and outliers by category")

plt.show()

print("\n2️⃣ VIOLIN PLOT - Distribution Shape:")
print("-" * 80)

# sns.violinplot() - Shows distribution shape (like box plot + KDE)
# SYNTAX: sns.violinplot(data=df, x='category', y='value')
# - Shows full distribution shape, not just quartiles
# - Wider sections = more data points

plt.figure(figsize=(10, 6))
sns.violinplot(data=orders, x='Category', y='TotalAmount', 
               palette='muted', inner='box')
plt.title('Order Amount Distribution by Category\n(Violin Plot)', 
         fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Total Amount ($)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_violinplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_violinplot.png")
print("   📊 Shows: Full distribution shape for each category")

plt.show()

print("\n3️⃣ BARPLOT - Statistical Bar Chart:")
print("-" * 80)

# sns.barplot() - Creates bar chart with confidence intervals
# SYNTAX: sns.barplot(data=df, x='category', y='value')
# - Automatically calculates mean and confidence intervals
# - Shows error bars (statistical significance)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Average order amount by category
sns.barplot(data=orders, x='Category', y='TotalAmount', 
            palette='viridis', ax=axes[0], ci='sd')  # ci='sd' shows std dev
axes[0].set_title('Average Order Amount by Category\n(With Error Bars)', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Average Amount ($)', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Right: Bar plot with hue
sns.barplot(data=orders, x='Category', y='TotalAmount', 
            hue='Shipped', palette='coolwarm', ax=axes[1])
axes[1].set_title('Average Order Amount by Category and Shipping', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Average Amount ($)', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Shipped', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_barplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_barplot.png")
print("   📊 Shows: Statistical comparisons with confidence intervals")

plt.show()

print("\n4️⃣ COUNTPLOT - Counting Categories:")
print("-" * 80)

# sns.countplot() - Counts occurrences of each category
# SYNTAX: sns.countplot(data=df, x='category')
# - Similar to value_counts() but as a plot

plt.figure(figsize=(10, 6))
sns.countplot(data=orders, x='Category', palette='Set3', order=orders['Category'].value_counts().index)
plt.title('Number of Orders by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)

# Add count labels on bars
for container in plt.gca().containers:
    plt.gca().bar_label(container, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_countplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_countplot.png")
print("   📊 Shows: Frequency of each category")

plt.show()

# ============================================================================
# STEP 4: RELATIONSHIP PLOTS - Finding Relationships
# ============================================================================

print("\n" + "="*80)
print("STEP 4: RELATIONSHIP PLOTS - Finding Relationships Between Variables")
print("="*80)

# WHY RELATIONSHIP PLOTS?
# - Identify correlations between features
# - Find patterns and trends
# - Guide feature engineering (which features to combine)

print("\n1️⃣ SCATTERPLOT - Two-Variable Relationships:")
print("-" * 80)

# sns.scatterplot() - Scatter plot with automatic grouping
# SYNTAX: sns.scatterplot(data=df, x='var1', y='var2', hue='category')
# - Can automatically color by category
# - Great for finding correlations

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Basic scatter plot
sns.scatterplot(data=orders, x='Price', y='Quantity', 
                size='TotalAmount', hue='Category', 
                sizes=(50, 300), alpha=0.7, ax=axes[0])
axes[0].set_title('Price vs Quantity\n(Size = Total Amount, Color = Category)', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Price ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Quantity', fontsize=11, fontweight='bold')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Right: Scatter plot with regression line
sns.scatterplot(data=orders, x='Price', y='TotalAmount', 
                hue='Category', alpha=0.6, ax=axes[1])
sns.regplot(data=orders, x='Price', y='TotalAmount', 
           scatter=False, color='red', ax=axes[1])  # Add regression line
axes[1].set_title('Price vs Total Amount\n(With Trend Line)', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Price ($)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total Amount ($)', fontsize=11, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_scatterplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_scatterplot.png")
print("   📊 Shows: Relationships between numeric variables")
print("   💡 Feature Engineering: Identify feature interactions")

plt.show()

print("\n2️⃣ LINEPLOT - Trends Over Time:")
print("-" * 80)

# sns.lineplot() - Line plot with confidence intervals
# SYNTAX: sns.lineplot(data=df, x='time', y='value', hue='category')
# - Automatically calculates confidence intervals
# - Great for time series data

# Prepare time series data
daily_data = orders.groupby(['OrderDate', 'Category'])['TotalAmount'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_data, x='OrderDate', y='TotalAmount', 
             hue='Category', marker='o', linewidth=2.5, markersize=8)
plt.title('Daily Revenue Trend by Category\n(With Confidence Intervals)', 
         fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Revenue ($)', fontsize=12, fontweight='bold')
plt.legend(title='Category', fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_lineplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_lineplot.png")
print("   📊 Shows: Trends over time with statistical confidence")

plt.show()

print("\n3️⃣ RELPLOT - Multi-Panel Relationship Plot:")
print("-" * 80)

# sns.relplot() - Flexible relationship plot (can create subplots)
# SYNTAX: sns.relplot(data=df, x='var1', y='var2', col='category', kind='scatter')
# - Can create separate plots for each category
# - kind: 'scatter' or 'line'

g = sns.relplot(data=orders, x='Price', y='TotalAmount', 
                col='Category', hue='Shipped', 
                kind='scatter', col_wrap=2, height=4, aspect=1.2, alpha=0.7)
g.fig.suptitle('Price vs Total Amount by Category', 
               fontsize=14, fontweight='bold', y=1.02)
g.set_axis_labels('Price ($)', 'Total Amount ($)', fontweight='bold')
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_relplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_relplot.png")
print("   📊 Shows: Relationships split by category")

plt.show()

# ============================================================================
# STEP 5: REGRESSION PLOTS - Trend Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 5: REGRESSION PLOTS - Trend Analysis")
print("="*80)

# WHY REGRESSION PLOTS?
# - Identify linear relationships
# - Understand trends and patterns
# - Guide feature engineering (polynomial features, interactions)

print("\n1️⃣ REGPLOT - Regression Plot with Confidence Interval:")
print("-" * 80)

# sns.regplot() - Scatter plot with regression line and confidence interval
# SYNTAX: sns.regplot(data=df, x='var1', y='var2', order=2)
# - order: Polynomial order (1=linear, 2=quadratic, etc.)
# - Shows confidence interval around regression line

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Linear regression
sns.regplot(data=orders, x='Price', y='TotalAmount', 
           scatter_kws={'alpha': 0.6, 'color': 'steelblue'}, 
           line_kws={'color': 'red', 'linewidth': 2}, ax=axes[0])
axes[0].set_title('Linear Regression: Price vs Total Amount', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Price ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Total Amount ($)', fontsize=11, fontweight='bold')

# Right: Polynomial regression (order=2)
sns.regplot(data=orders, x='Price', y='TotalAmount', order=2,
           scatter_kws={'alpha': 0.6, 'color': 'steelblue'}, 
           line_kws={'color': 'green', 'linewidth': 2}, ax=axes[1])
axes[1].set_title('Polynomial Regression (Order=2): Price vs Total Amount', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Price ($)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total Amount ($)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_regplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_regplot.png")
print("   📊 Shows: Linear and polynomial relationships")
print("   💡 Feature Engineering: Identify if polynomial features needed")

plt.show()

print("\n2️⃣ LMPLOT - Regression Plot with Grouping:")
print("-" * 80)

# sns.lmplot() - Regression plot with grouping (hue, col, row)
# SYNTAX: sns.lmplot(data=df, x='var1', y='var2', hue='category')
# - Can create separate regression lines for each group

g = sns.lmplot(data=orders, x='Price', y='TotalAmount', 
              hue='Category', col='Category', col_wrap=2, 
              height=4, aspect=1.2, scatter_kws={'alpha': 0.6})
g.fig.suptitle('Regression Analysis: Price vs Total Amount by Category', 
               fontsize=14, fontweight='bold', y=1.02)
g.set_axis_labels('Price ($)', 'Total Amount ($)', fontweight='bold')
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_lmplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_lmplot.png")
print("   📊 Shows: Regression relationships by category")

plt.show()

# ============================================================================
# STEP 6: HEATMAPS - Correlation and Multi-dimensional Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 6: HEATMAPS - Correlation and Multi-dimensional Analysis")
print("="*80)

# WHY HEATMAPS?
# - Visualize correlations between features
# - Identify redundant features (high correlation)
# - Guide feature selection for ML

print("\n1️⃣ CORRELATION HEATMAP:")
print("-" * 80)

# sns.heatmap() - Creates heatmap from correlation matrix
# SYNTAX: sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# - annot: Show correlation values
# - cmap: Color map
# - vmin, vmax: Value range for colors

# Calculate correlation matrix
numeric_cols = orders.select_dtypes(include=[np.number]).columns
correlation_matrix = orders[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8},
           fmt='.2f', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap\n(For Feature Engineering)', 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_correlation_heatmap.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_correlation_heatmap.png")
print("   📊 Shows: Correlations between all numeric features")
print("   💡 Feature Engineering: Remove highly correlated features (>0.9)")

plt.show()

print("\n2️⃣ PIVOT TABLE HEATMAP:")
print("-" * 80)

# Heatmap from pivot table (multi-dimensional analysis)
pivot_data = orders.pivot_table(
    values='TotalAmount',
    index='Category',
    columns='OrderMonth',
    aggfunc='sum',
    fill_value=0
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
           linewidths=0.5, cbar_kws={'label': 'Revenue ($)'})
plt.title('Revenue Heatmap: Category × Month\n(Feature Engineering: Interaction Features)', 
         fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Category', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_pivot_heatmap.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_pivot_heatmap.png")
print("   📊 Shows: Multi-dimensional patterns (Category × Month)")

plt.show()

# ============================================================================
# STEP 7: PAIR PLOT - Multi-Variable Comparison
# ============================================================================

print("\n" + "="*80)
print("STEP 7: PAIR PLOT - Multi-Variable Comparison")
print("="*80)

# WHY PAIR PLOTS?
# - Compare all pairs of variables at once
# - Identify relationships across multiple features
# - Great for feature engineering (which features are related)

print("\n1️⃣ PAIRPLOT - All Pairs Scatter Plot:")
print("-" * 80)

# sns.pairplot() - Creates scatter plots for all variable pairs
# SYNTAX: sns.pairplot(data=df, vars=['col1', 'col2', ...], hue='category')
# - vars: Which columns to include
# - hue: Color by category
# - diag_kind: 'hist' or 'kde' for diagonal plots

# Select key numeric features
feature_cols = ['Price', 'Quantity', 'TotalAmount', 'OrderMonth', 'OrderDayOfWeek']

# Create pair plot (subset of data for clarity)
sample_data = orders[feature_cols + ['Category']].sample(min(100, len(orders)), random_state=42)

g = sns.pairplot(sample_data, vars=feature_cols, hue='Category', 
                diag_kind='kde', plot_kws={'alpha': 0.6}, 
                height=2.5, aspect=1)
g.fig.suptitle('Pair Plot - Feature Relationships\n(All Pairs Comparison)', 
               fontsize=14, fontweight='bold', y=1.02)
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_pairplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_pairplot.png")
print("   📊 Shows: All pairwise relationships between features")
print("   💡 Feature Engineering: Identify which features to combine")

plt.show()

# ============================================================================
# STEP 8: ADVANCED FEATURES - FacetGrid and Customization
# ============================================================================

print("\n" + "="*80)
print("STEP 8: ADVANCED FEATURES - FacetGrid and Customization")
print("="*80)

# WHY FACETGRID?
# - Create multiple plots based on categories
# - Compare patterns across groups
# - Customize each subplot individually

print("\n1️⃣ FACETGRID - Custom Multi-Panel Plots:")
print("-" * 80)

# sns.FacetGrid() - Create grid of plots
# SYNTAX: g = sns.FacetGrid(df, col='category', row='category2')
#         g.map(plt.plot_type, 'x', 'y')

# Create FacetGrid
g = sns.FacetGrid(orders, col='Category', col_wrap=2, height=4, aspect=1.2)
g.map_dataframe(sns.histplot, x='TotalAmount', kde=True, bins=15)
g.set_axis_labels('Total Amount ($)', 'Frequency', fontweight='bold')
g.fig.suptitle('Distribution of Order Amounts by Category\n(FacetGrid)', 
               fontsize=14, fontweight='bold', y=1.02)
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_facetgrid.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_facetgrid.png")
print("   📊 Shows: Multiple plots in a grid layout")

plt.show()

print("\n2️⃣ CUSTOM STYLING - Beautiful Plots:")
print("-" * 80)

# Customize seaborn style
sns.set_style("darkgrid")
sns.set_palette("husl")
sns.set_context("notebook")  # Options: "paper", "notebook", "talk", "poster"

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=orders, x='Category', y='TotalAmount', 
                palette='muted')
plt.title('Custom Styled Box Plot', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Total Amount ($)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)

# Add statistical annotations
for i, category in enumerate(orders['Category'].unique()):
    category_data = orders[orders['Category'] == category]['TotalAmount']
    median = category_data.median()
    plt.text(i, median, f'${median:.0f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_custom_style.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_custom_style.png")
print("   📊 Shows: Custom styling and annotations")

plt.show()

# Reset style
sns.set_style("whitegrid")

# ============================================================================
# STEP 9: FEATURE ENGINEERING VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: FEATURE ENGINEERING VISUALIZATIONS")
print("="*80)

# Visualizing engineered features to validate their usefulness

print("\n1️⃣ CUSTOMER SEGMENTATION VISUALIZATION:")
print("-" * 80)

# Visualize customer features
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Customer order count distribution
sns.histplot(data=customer_features, x='OrderCount', kde=True, 
            bins=15, color='purple', ax=axes[0])
axes[0].set_title('Customer Order Count Distribution', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Orders', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Number of Customers', fontsize=11, fontweight='bold')

# Right: Total Spent vs Order Count
sns.scatterplot(data=customer_features, x='OrderCount', y='TotalSpent',
               size='AvgOrderValue', hue='AvgOrderValue', 
               sizes=(50, 300), palette='viridis', alpha=0.7, ax=axes[1])
axes[1].set_title('Customer Value Analysis\n(Size & Color = Avg Order Value)', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Orders', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total Spent ($)', fontsize=11, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_customer_features.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_customer_features.png")
print("   📊 Shows: Engineered customer features for segmentation")

plt.show()

print("\n2️⃣ TIME-BASED FEATURE ANALYSIS:")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time-Based Feature Engineering Analysis', 
             fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Revenue by Day of Week
dow_data = orders.groupby('OrderDayOfWeek')['TotalAmount'].sum().reset_index()
sns.barplot(data=dow_data, x='OrderDayOfWeek', y='TotalAmount', 
           palette='coolwarm', ax=axes[0, 0])
axes[0, 0].set_title('Revenue by Day of Week', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day of Week (0=Mon, 6=Sun)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Total Revenue ($)', fontsize=11, fontweight='bold')

# Plot 2: Average order by month
month_data = orders.groupby('OrderMonth')['TotalAmount'].mean().reset_index()
sns.lineplot(data=month_data, x='OrderMonth', y='TotalAmount', 
            marker='o', linewidth=2.5, markersize=10, ax=axes[0, 1])
axes[0, 1].set_title('Average Order Value by Month', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Month', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Average Amount ($)', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Category performance over time
category_month = orders.groupby(['OrderMonth', 'Category'])['TotalAmount'].sum().reset_index()
sns.lineplot(data=category_month, x='OrderMonth', y='TotalAmount', 
            hue='Category', marker='o', linewidth=2, ax=axes[1, 0])
axes[1, 0].set_title('Category Revenue Trend', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Month', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Total Revenue ($)', fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=9)

# Plot 4: Day of week distribution by category
sns.boxplot(data=orders, x='Category', y='OrderDayOfWeek', 
           palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title('Order Day of Week by Category', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Day of Week', fontsize=11, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_time_features.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_time_features.png")
print("   📊 Shows: Time-based feature patterns")

plt.show()

# ============================================================================
# STEP 10: STATISTICAL SUMMARY VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: STATISTICAL SUMMARY VISUALIZATIONS")
print("="*80)

print("\n1️⃣ CATPLOT - Categorical Plot (Flexible):")
print("-" * 80)

# sns.catplot() - Flexible categorical plot (can create different plot types)
# SYNTAX: sns.catplot(data=df, x='cat', y='val', kind='violin', col='category')
# - kind: 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count'

g = sns.catplot(data=orders, x='Category', y='TotalAmount', 
               kind='violin', hue='Shipped', col='Shipped',
               height=5, aspect=1.2, palette='pastel')
g.fig.suptitle('Order Amount Distribution: Category × Shipped Status', 
               fontsize=14, fontweight='bold', y=1.02)
g.set_axis_labels('Category', 'Total Amount ($)', fontweight='bold')
g.set_xticklabels(rotation=45)
plt.savefig('/Users/satya/PythonLearn/PythonBasics/seaborn_catplot.png', 
            dpi=150, bbox_inches='tight')
print("   ✅ Created: seaborn_catplot.png")
print("   📊 Shows: Flexible categorical plotting")

plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL SUMMARY - SEABORN VISUALIZATION COMPLETE")
print("="*80)

print(f"""
✅ VISUALIZATIONS CREATED:

1. 📉 Distribution Plots (histplot, displot)
   - Files: seaborn_distribution.png, seaborn_kde.png
   - Purpose: Understand data distributions

2. 📦 Categorical Plots (boxplot, violinplot, barplot, countplot)
   - Files: seaborn_boxplot.png, seaborn_violinplot.png, seaborn_barplot.png, seaborn_countplot.png
   - Purpose: Compare categories statistically

3. 🔍 Relationship Plots (scatterplot, lineplot, relplot)
   - Files: seaborn_scatterplot.png, seaborn_lineplot.png, seaborn_relplot.png
   - Purpose: Find relationships between variables

4. 📈 Regression Plots (regplot, lmplot)
   - Files: seaborn_regplot.png, seaborn_lmplot.png
   - Purpose: Identify trends and linear relationships

5. 🔥 Heatmaps (correlation, pivot tables)
   - Files: seaborn_correlation_heatmap.png, seaborn_pivot_heatmap.png
   - Purpose: Multi-dimensional analysis and correlation

6. 🔗 Pair Plot
   - File: seaborn_pairplot.png
   - Purpose: Compare all variable pairs

7. 🎨 Advanced Features (FacetGrid, custom styling)
   - Files: seaborn_facetgrid.png, seaborn_custom_style.png
   - Purpose: Custom multi-panel plots

8. 👥 Feature Engineering Visualizations
   - Files: seaborn_customer_features.png, seaborn_time_features.png
   - Purpose: Validate engineered features

9. 📊 Statistical Summaries (catplot)
   - File: seaborn_catplot.png
   - Purpose: Flexible categorical analysis

📚 KEY SEABORN FUNCTIONS LEARNED:

DISTRIBUTION PLOTS:
   ✅ sns.histplot() - Histogram with KDE
   ✅ sns.displot() - Flexible distribution plot
   ✅ sns.kdeplot() - Kernel density estimate

CATEGORICAL PLOTS:
   ✅ sns.boxplot() - Box and whisker plot
   ✅ sns.violinplot() - Violin plot (distribution shape)
   ✅ sns.barplot() - Statistical bar chart
   ✅ sns.countplot() - Count occurrences
   ✅ sns.catplot() - Flexible categorical plot

RELATIONSHIP PLOTS:
   ✅ sns.scatterplot() - Scatter plot
   ✅ sns.lineplot() - Line plot with confidence intervals
   ✅ sns.relplot() - Multi-panel relationship plot

REGRESSION PLOTS:
   ✅ sns.regplot() - Regression plot
   ✅ sns.lmplot() - Regression plot with grouping

HEATMAPS:
   ✅ sns.heatmap() - Heatmap visualization

MULTI-VARIABLE:
   ✅ sns.pairplot() - Pair plot (all pairs)

ADVANCED:
   ✅ sns.FacetGrid() - Custom multi-panel plots
   ✅ sns.set_style() - Set plot style
   ✅ sns.set_palette() - Set color palette

💡 FEATURE ENGINEERING INSIGHTS FROM SEABORN:

1. ✅ Distribution analysis → Identify transformation needs (log, sqrt)
2. ✅ Categorical comparisons → Feature importance by category
3. ✅ Relationship plots → Identify feature interactions
4. ✅ Regression plots → Identify polynomial features needed
5. ✅ Correlation heatmap → Remove redundant features
6. ✅ Pair plots → Find feature combinations
7. ✅ Time-based analysis → Time series features
8. ✅ Customer segmentation → Customer-based features

🎯 SEABORN ADVANTAGES OVER MATPLOTLIB:

✅ Automatic statistical aggregations (mean, confidence intervals)
✅ Beautiful default styles and color palettes
✅ Easy grouping with hue, col, row parameters
✅ Built-in regression and trend lines
✅ Better default aesthetics
✅ Works seamlessly with pandas DataFrames
✅ Less code for same visualizations

🎯 WHEN TO USE SEABORN:

✅ Statistical visualizations
✅ Comparing groups/categories
✅ Relationship analysis
✅ Distribution analysis
✅ Quick, beautiful plots
✅ When you need confidence intervals
✅ When you need regression lines

🎯 WHEN TO USE MATPLOTLIB:

✅ Full control over plot elements
✅ Custom, complex visualizations
✅ Fine-grained customization
✅ When seaborn doesn't have the plot type you need
✅ Publication-quality figures with specific requirements

""")

print("="*80)
print("✅ SEABORN TUTORIAL COMPLETE!")
print("="*80)
print("\n💡 TIP: Seaborn makes statistical visualization easy and beautiful!")
print("   Combine with pandas for powerful data analysis workflows.")
print("="*80)

