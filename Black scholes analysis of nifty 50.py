import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Read the CSV file and specify data types for relevant columns
dtype_dict = {
    'Underlying_Price': float,  # We'll read this as a string to handle commas
    'Strike Price': float,
    'Risk-Free Interest Rate': float,
    'Volatility': float,
    'Time to Expiration': float,
    'LTP':float
}
option_data = pd.read_csv('option-chain-ED-NIFTY-31-Aug-2023.csv', dtype=dtype_dict)

def black_scholes(option_type, S, K, r, sigma, T):
    d1 = (np.log(float(S) / float(K)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'CALL':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return option_price

option_data['BS_Price'] = option_data.apply(
    lambda row: black_scholes(row['Option Type'], row['Underlying Price'], row['Strike Price'],
    row['Risk-Free Interest Rate'], row['Volatility'], row['Time to Expiry']),
    
    axis = 1
)

# Calculate absolute difference between Black-Scholes and real prices
option_data['Price_Difference'] = np.abs(option_data['BS_Price'] - option_data['LTP'])

# Calculate mean absolute difference
mean_abs_difference = option_data['Price_Difference'].mean()

print("Mean Absolute Difference:", mean_abs_difference)

call = option_data[(option_data['Option Type'] == 'CALL')]
put = option_data[(option_data['Option Type'] == 'PUT')]


plt.figure(figsize=(12,6))
plt.plot(call['Strike Price'], call['LTP'],label='LTP call', marker='o',color = 'pink')
plt.plot(call['Strike Price'], call['BS_Price'], label='Real Prices call', marker='x',color = 'purple')
plt.plot(put['Strike Price'], put['LTP'],label='LTP put', marker='^', color = 'blue')
plt.plot(put['Strike Price'], put['BS_Price'], label='Real Prices put', marker='s',color = 'green')


plt.xlabel('Strike price')
plt.ylabel('Stock prices (LTP and Real prices)') 
plt.title('Comparison of Real Prices and LTP for Nifty 50')


# Add a legend
plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
