import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24
# Load the DataFrame from a CSV file.
# Make sure to update the file path according to your CSV structure.
df = pd.read_csv("result.csv")

# Check column names to ensure they are correct
print(df.columns)

# Calculate Mean Squared Error (MSE) between Predicted_Label and Actual_Label
df['MSE_PARIS'] = (df['Obs_SDC_Norm'] - df['PARIS_SDC_Norm']) ** 2
df['MSE_LSTM'] = (df['Predicted Label'] - df['Actual Label']) ** 2
# Sort data by MSE
df_sorted = df.sort_values(by='wordCount')

# Create a plot
fig, ax1 = plt.subplots(figsize=(20, 13))
index = range(len(df['MSE_PARIS']))
bar_width = 0.35
opacity = 0.8
# Histogram for MSE
rects1 = ax1.bar(index, df_sorted['MSE_LSTM'], bar_width, alpha=opacity, color='orange', label='LSTM')
rects2 = ax1.bar([p + bar_width for p in index], df_sorted['MSE_PARIS'], bar_width, alpha=opacity, color='grey', label='PARIS')
ax1.set_xlabel('BenchMark')
ax1.set_ylabel('MSE')
ax1.set_xticks([p + bar_width/2 for p in index])
#ax1.set_xticks(df_sorted['BenchMark'])
ax1.set_xticklabels(df_sorted['BenchMark'], rotation=45, ha="right")

# Line chart for wordCount on a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(df_sorted['BenchMark'], df_sorted['wordCount'], color='green', marker='o', label='Word Count')  # Make sure this matches exactly
ax2.set_ylabel('Word Count')

# Compute the average MSE
avg_mse_PARIS = df['MSE_PARIS'].mean()
avg_mse_LSTM = df['MSE_LSTM'].mean()

# Add a line for the average MSE
ax1.axhline(avg_mse_PARIS, color='grey', linestyle='--', label='Average MSE of PARIS')
ax1.axhline(avg_mse_LSTM, color='orange', linestyle='--', label='Average MSE of LSTM')

# Add custom text regarding the CC_MSE_WC value
#plt.figtext(0.5, 0.8, "CC_MSE_WC = 0.512", ha="center", fontsize=24, color='black')


# Legend
fig.legend(loc='center', bbox_to_anchor=(0.5, 0.8))

# Removing grid lines
ax1.grid(False)
ax2.grid(True)
plt.subplots_adjust(top=0.99, 
                    bottom=0.13, 
                    left=0.06, 
                    right=0.9, 
                    hspace=0.01, 
                    wspace=0.01)
# Save the plot as PDF
plt.savefig("wc_LSTM_PARIS.png")

plt.show()