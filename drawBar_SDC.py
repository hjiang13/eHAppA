import matplotlib.pyplot as plt

# Data for both benchmarks
lstm_benchmark = ["IS", "LU", "Nn", "Myocyte", "Backprop", "CG", "MG", "BT", "SP", "DC", "Lud", "Kmeans", "STREAM", "Blackscholes", "PuReMD", "Lulesh", "Hotspot", "Bfs-rodinia", "Nw", "Pathfinder"]
lstm_mse = [0.000361, 0.095481, 0.103041, 0.0361, 0.056644, 0.0441, 0.047089, 0.092416, 0.609961, 0.007056, 0.044944, 0.013924, 0.000729, 0.023409, 0.002601, 0.021609, 0.024649, 0.027225, 0.0441, 0.265225]
paris_benchmark = ["SP", "LU", "MG", "Myocyte", "Lud", "CG", "IS", "DC", "PuReMD", "Bfs-rodinia", "Kmeans", "Nw", "Backprop", "Blackscholes", "BT", "Hotspot", "Pathfinder", "Nn", "STREAM", "Lulesh"]
paris_mse = [0.0001, 0.0006, 0.0012, 0.0018, 0.0054, 0.0168, 0.0217, 0.0303, 0.0375, 0.0412, 0.0559, 0.0751, 0.1626, 0.1631, 0.2059, 0.2318, 0.2860, 0.3069, 0.3102, 0.3894]

# Sort the data based on LSTM MSE
sorted_indices = sorted(range(len(lstm_mse)), key=lambda k: lstm_mse[k])
sorted_lstm_benchmark = [lstm_benchmark[i] for i in sorted_indices]
sorted_lstm_mse = [lstm_mse[i] for i in sorted_indices]
sorted_paris_mse = [paris_mse[paris_benchmark.index(sorted_lstm_benchmark[i])] for i in sorted_indices]

# Create the bar plot
fig, ax = plt.subplots()
index = range(len(sorted_lstm_benchmark))
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index, sorted_lstm_mse, bar_width, alpha=opacity, color='orange', label='LSTM')
rects2 = ax.bar([p + bar_width for p in index], sorted_paris_mse, bar_width, alpha=opacity, color='grey', label='PARIS')

ax.set_xlabel('Benchmark')
ax.set_ylabel('MSE')
ax.set_xticks([p + bar_width/2 for p in index])
ax.set_xticklabels(sorted_lstm_benchmark, rotation=75)
ax.legend()

plt.show()
plt.subplots_adjust(top=0.925, 
                    bottom=0.20, 
                    left=0.10, 
                    right=0.99, 
                    hspace=0.01, 
                    wspace=0.01)
plt.savefig("bar_SDC_LSTMvsPARIS.png")