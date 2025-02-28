import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)
#Rename columns for consistency
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour Participation Rate',
    'Date': 'Date'
}, inplace=True)

#Check if 'Date' column exists and fix formatting
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
else:
    print("❌ Error: 'Date' column not found! Check dataset columns.")

#Remove rows with missing values
df.dropna(inplace=True)
#Ensure 'Unemployment Rate' exists
if 'Unemployment Rate' not in df.columns:
    print("❌ Error: 'Unemployment Rate' column not found! Check dataset columns.")
else:
    # Histogram of Unemployment Rate
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Unemployment Rate'], bins=30, kde=True, color="blue")
    plt.xlabel("Unemployment Rate (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Unemployment Rate in India")
    plt.show(block=True)

    # Line plot of Unemployment Rate over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Unemployment Rate", marker="o", color="red")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.title("Unemployment Rate Over Time in India")
    plt.xticks(rotation=45)
    plt.show(block=True)

    # Bar plot of Unemployment Rate by Region
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Region", y="Unemployment Rate", hue="Region", dodge=False, legend=False)
    plt.xlabel("Region")
    plt.ylabel("Unemployment Rate (%)")
    plt.title("Unemployment Rate by Region in India")
    plt.xticks(rotation=90)
    plt.show(block=True)

#Save the cleaned dataset
df.to_csv("Cleaned_Unemployment_Data.csv", index=False)
print("\n Cleaned dataset saved successfully.")








