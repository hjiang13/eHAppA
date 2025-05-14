import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
# Load the DataFrame from a CSV file.
# Make sure to update the file path according to your CSV structure.
df = pd.read_csv("result.csv")

# Check column names to ensure they are correct
print(df.columns)

# Calculate Mean Squared Error (MSE) between Predicted_Label and Actual_Label
df['MSE'] = (df['Predicted Label'] - df['Actual Label']) ** 2

# Sort data by MSE
df_sorted = df.sort_values(by='wordCount')

# Create a plot
fig, ax1 = plt.subplots(figsize=(20, 13))

# Histogram for MSE
ax1.bar(df_sorted['BenchMark'], df_sorted['MSE'], color="orange", label='MSE')
ax1.set_xlabel('BenchMark')
ax1.set_ylabel('MSE')
ax1.set_xticks(df_sorted['BenchMark'])
ax1.set_xticklabels(df_sorted['BenchMark'], rotation=45, ha="right")

# Line chart for wordCount on a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(df_sorted['BenchMark'], df_sorted['wordCount'], color='green', marker='o', label='Word Count')  # Make sure this matches exactly
ax2.set_ylabel('Word Count')

# Compute the average MSE
avg_mse = df['MSE'].mean()

# Add a line for the average MSE
ax1.axhline(avg_mse, color='red', linestyle='--', label='Average MSE')

# Add custom text regarding the CC_MSE_WC value
plt.figtext(0.5, 0.8, "CC_MSE_WC = 0.097", ha="center", fontsize=24, color='black')


# Legend
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Removing grid lines
ax1.grid(False)
ax2.grid(True)

# Save the plot as PDF
plt.savefig("LSTM_SDC.png")

plt.show()