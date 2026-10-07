import pandas as pd

df_customers = pd.DataFrame({
    'Customer_ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 34, 45],
    'Country': ['USA', 'UK', 'USA']
})

df_transactions = pd.DataFrame({
    'Transaction_ID': [101, 102, 103, 104],
    'Customer_ID': [1, 2, 1, 3],
    'Transaction_Amount': [200, 150, 300, 400],
    'Transaction_Date': ['2023-01-01', '2023-01-02', '2023-01-05', '2023-01-08']
})

merged_df = pd.merge(df_customers, df_transactions, on='Customer_ID', how='inner')

avg_amount_per_country = merged_df.groupby('Country')['Transaction_Amount'].mean().reset_index()

avg_amount_per_country.rename(columns={'Transaction_Amount': 'Average_Transaction_Amount'}, inplace=True)

print(avg_amount_per_country)