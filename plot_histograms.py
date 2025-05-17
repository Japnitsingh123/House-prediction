import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (make sure the CSV file is in the same folder or give full path)
df = pd.read_csv('train.csv')

# List of features you want to plot
features = ['YearBuilt', 'OverallQual', 'GarageCars', 'TotRmsAbvGrd', 'Fireplaces', 'LotArea']

# Plot histograms
df[features].hist(bins=15, figsize=(15,10), color='blue')
plt.suptitle('Histograms of Features in model', fontsize=16)

# Save the figure as a file (optional)
plt.savefig('histograms.png')

# Show the plots
plt.show()
