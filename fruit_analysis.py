import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("fruit_vegetable_sales.csv")

# Display dataset
print("FRUIT AND VEGETABLE SALES DATA")
print(df)

# Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Total sales
print("\nTOTAL SALES:", df["Total_Sales"].sum())

# Average sales
print("AVERAGE SALES:", df["Total_Sales"].mean())

# Best-selling product
print("\nBEST-SELLING PRODUCT")
print(df.loc[df["Units_Sold"].idxmax()])

# Lowest-selling product
print("\nLOWEST-SELLING PRODUCT")
print(df.loc[df["Units_Sold"].idxmin()])

# Sales by category
print("\nSALES BY CATEGORY")
print(df.groupby("Category")["Total_Sales"].sum())

# Sales by month
print("\nSALES BY MONTH")
print(df.groupby("Month")["Total_Sales"].sum())

# Correlation
print("\nCORRELATION")
print(df["Units_Sold"].corr(df["Total_Sales"]))

# Scatter plot
plt.scatter(df["Units_Sold"], df["Total_Sales"])
plt.xlabel("Units Sold")
plt.ylabel("Total Sales")
plt.title("Units Sold vs Total Sales")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare data
X = df[["Units_Sold", "Unit_Price"]]
y = df["Total_Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display results
print("\nW04 - LINEAR REGRESSION")
print("Training data:", len(X_train))
print("Testing data:", len(X_test))
print("Actual values:", list(y_test))
print("Predicted values:", y_pred)
print("TEST SET R2 SCORE:", r2_score(y_test, y_pred))