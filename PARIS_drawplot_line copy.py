import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
# Data
data = {

    "BenchMark":["SP", "LU", "MG", "Myocyte", "Lud", "CG", "IS", "DC", "PuReMD", "Bfs-rodinia", "Kmeans", "Nw", "Backprop", "Blackscholes", "BT", "Hotspot", "Pathfinder", "Nn", "STREAM", "Lulesh"],
    "benign MSE":[0.2071, 0.0304, 0.1503, 0.0197, 0.0749, 0.0207, 0.0020, 0.0220, 0.1697, 0.0003, 0.1122, 0.0411, 0.1941, 0.1023, 0.0480, 0.0111, 0.1036, 0.4476, 0.1174, 0.3318],
    "crash MSE":[0.2088, 0.0912, 0.1302, 0.0049, 0.0236, 0.0010, 0.0422, 0.0273, 0.0000, 0.0028, 0.0153, 0.0034, 0.0278, 0.0001, 0.2369, 0.5635, 0.0070, 0.0032, 0.0084, 0.0175],
    "SDC MSE":[0.0001, 0.0006, 0.0012, 0.0018, 0.0054, 0.0168, 0.0217, 0.0303, 0.0375, 0.0412, 0.0559, 0.0751, 0.1626, 0.1631, 0.2059, 0.2318, 0.2860, 0.3069, 0.3102, 0.3894]
}

# Create DataFrame and sort by SDC MSE
df = pd.DataFrame(data)
df_sorted = df.sort_values("SDC MSE")

# Plotting
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(df_sorted["BenchMark"], df_sorted["benign MSE"], label="Benign MSE", marker='v',color='blue')
ax.plot(df_sorted["BenchMark"], df_sorted["crash MSE"], label="Crash MSE", marker='o',color='orange')
ax.plot(df_sorted["BenchMark"], df_sorted["SDC MSE"], label="SDC MSE", marker='s',color='green')

# Calculate and plot averages for each MSE category
avg_benign_mse = df["benign MSE"].mean()
avg_crash_mse = df["crash MSE"].mean()
avg_sdc_mse = df["SDC MSE"].mean()

ax.axhline(y=avg_benign_mse, color='blue', linestyle='--', label=f"Average Benign MSE: {avg_benign_mse:.4f}")
ax.axhline(y=avg_crash_mse, color='orange', linestyle='--', label=f"Average Crash MSE: {avg_crash_mse:.4f}")
ax.axhline(y=avg_sdc_mse, color='green', linestyle='--', label=f"Average SDC MSE: {avg_sdc_mse:.4f}")

# Table data
table_data=[
    ["Benign MSE", "1.000", "-1.807", "0.536"],
    ["Crash MSE", "-1.807", "1.000", "0.103"],
    ["SDC MSE", "0.536", "0.103", "1.000"]
]

# Add table to the plot
table = ax.table(cellText=table_data, colLabels=["CC", "Benign MSE", "Crash MSE", "SDC MSE"], loc='upper center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(24)
table.scale(0.8, 2.4)  # Adjust scale to match the plot aesthetics

# Additional plot settings
ax.set_xlabel("Benchmark")
ax.set_ylabel("Mean Squared Error (MSE)")
#ax.set_title("MSE Comparison Across Benchmarks")
ax.legend(loc='center left')
ax.grid(axis='y')
plt.xticks(rotation=45)
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, wspace=0, hspace=0)
plt.show()
# Save the plot as PDF
plt.savefig("PARIS_all_line.png")
