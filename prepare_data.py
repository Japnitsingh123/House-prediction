import pandas as pd
import joblib

# Load the CSV file
df = pd.read_csv('ames.housing.csv')

# Keep only the SalePrice column (or more if you want)
df = df[['SalePrice']]

# Save as a pickle file for your app
joblib.dump(df, 'ames_housing_data.pkl')

print("Pickle file saved successfully!")
