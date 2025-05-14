import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20
# Data
data = {
    "BenchMark": ["IS", "LU", "Nn", "Myocyte", "Backprop", "CG", "MG", "BT", "SP", "DC", "Lud", "Kmeans", "STREAM", "Blackscholes", "PuReMD", "Lulesh", "Hotspot", "Bfs-rodinia", "Nw", "Pathfinder"],
    "benign MSE": [0.009604, 0.001444, 0.032761, 0.081225, 0.031329, 0.073984, 0.113569, 0.0064, 0.2209, 0.0004, 0.105625, 0.251001, 0.048841, 0.013225, 0.281961, 0.003844, 0.042436, 0.008281, 0.015129, 0.0016],
    "crash MSE": [0.0081, 0.013924, 0.000361, 0.022201, 0.000625, 0.159201, 0.042025, 0.075076, 0.000036, 0.016129, 0.200704, 0.173056, 0.042436, 0.039204, 0.299209, 0.0169, 0.093636, 0.034969, 0.055696, 0.118336],
    "SDC MSE": [0.000361, 0.095481, 0.103041, 0.0361, 0.056644, 0.0441, 0.047089, 0.092416, 0.609961, 0.007056, 0.044944, 0.013924, 0.000729, 0.023409, 0.002601, 0.021609, 0.024649, 0.027225, 0.0441, 0.265225]
}

# Create DataFrame and sort by SDC MSE
df = pd.DataFrame(data)
df_sorted = df.sort_values("SDC MSE")

# Plotting
fig, ax = plt.subplots(figsize=(20, 10))

# Calculate and plot averages for each MSE category
avg_benign_mse = df["benign MSE"].mean()
avg_crash_mse = df["crash MSE"].mean()
avg_sdc_mse = df["SDC MSE"].mean()

# Create bar chart
width = 0.3
x = range(len(df_sorted))
ax.bar(x, df_sorted["benign MSE"], width, label='Benign MSE', color='green')
ax.bar([p + width for p in x], df_sorted["crash MSE"], width, label='Crash MSE', color='red')
ax.bar([p + 2*width for p in x], df_sorted["SDC MSE"], width, label='SDC MSE', color='orange')
# Table data
table_data=[
    ["Benign MSE", "1.000", "0.605", "0.258"],
    ["Crash MSE", "0.605", "1.000", "-0.194"],
    ["SDC MSE", "0.258", "-0.194", "1.000"]
]

# Add table to the plot
table = ax.table(cellText=table_data, colLabels=["CC", "Benign MSE", "Crash MSE", "SDC MSE"], loc='upper center', cellLoc='center', fontsize = 24)
table.auto_set_font_size(False)
table.set_fontsize(24)
table.scale(0.8, 2.4)  # Adjust scale to match the plot aesthetics

# Additional plot settings
ax.set_xlabel("Benchmark", fontsize=24)
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=24)
#ax.set_title("MSE Comparison Across Benchmarks")
ax.legend(loc='center left')
ax.grid(axis='y')
plt.xticks([p + width for p in x], df_sorted["BenchMark"], rotation=45)
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, wspace=0, hspace=0)
plt.show()
# Save the plot as PDF
plt.savefig("LSTM_all_bar.png")
