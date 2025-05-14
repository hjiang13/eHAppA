import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating the DataFrame
data = {
    "BenchMark": ["IS", "LU", "Nn", "Myocyte", "Backprop", "CG", "MG", "BT", "SP", "DC",
                  "Lud", "Kmeans", "STREAM", "Blackscholes", "PuReMD", "Lulesh", "Hotspot", "Bfs-rodinia", "Nw", "Pathfinder"],
    "benign MSE": [0.009604, 0.001444, 0.032761, 0.081225, 0.031329, 0.073984, 0.113569, 0.0064, 0.2209, 0.0004,
                   0.105625, 0.251001, 0.048841, 0.013225, 0.281961, 0.003844, 0.042436, 0.008281, 0.015129, 0.0016],
    "crash MSE": [0.0081, 0.013924, 0.000361, 0.022201, 0.000625, 0.159201, 0.042025, 0.075076, 0.000036, 0.016129,
                  0.200704, 0.173056, 0.042436, 0.039204, 0.299209, 0.0169, 0.093636, 0.034969, 0.055696, 0.118336],
    "SDC MSE": [0.000361, 0.095481, 0.103041, 0.0361, 0.056644, 0.0441, 0.047089, 0.092416, 0.609961, 0.007056,
                0.044944, 0.013924, 0.000729, 0.023409, 0.002601, 0.021609, 0.024649, 0.027225, 0.0441, 0.265225]
}

df = pd.DataFrame(data)
df = df.sort_values(by="SDC MSE", ascending=True)

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(df['BenchMark']))
width = 0.3

# Bars for MSE
rects1 = ax.bar(x - width, df['benign MSE'], width, label='Benign MSE',color='b')
rects2 = ax.bar(x, df['crash MSE'], width, label='Crash MSE',color='orange')
rects3 = ax.bar(x + width, df['SDC MSE'], width, label='SDC MSE',color='g')

# Lines for average MSE
benign_avg = df['benign MSE'].mean()
crash_avg = df['crash MSE'].mean()
sdc_avg = df['SDC MSE'].mean()

ax.axhline(y=benign_avg, color='b', linestyle='-', label=f'Avg Benign MSE: {benign_avg:.4f}')
ax.axhline(y=crash_avg, color='orange', linestyle='-', label=f'Avg Crash MSE: {crash_avg:.4f}')
ax.axhline(y=sdc_avg, color='g', linestyle='-', label=f'Avg SDC MSE: {sdc_avg:.4f}')

# Labels and settings
ax.set_xlabel('BenchMark')
ax.set_ylabel('MSE Values')
#ax.set_title('MSE Comparison by BenchMark')
ax.set_xticks(x)
ax.set_xticklabels(df['BenchMark'], rotation=45)
ax.legend()

# Remove grid lines
ax.grid(False)

plt.show()
# Save the plot as PDF
plt.savefig("LSTM_all.png")