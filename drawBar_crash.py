import matplotlib.pyplot as plt

# Data for both benchmarks
lstm_benchmark = ["IS", "LU", "Nn", "Myocyte", "Backprop", "CG", "MG", "BT", "SP", "DC", "Lud", "Kmeans", "STREAM", "Blackscholes", "PuReMD", "Lulesh", "Hotspot", "Bfs-rodinia", "Nw", "Pathfinder"]
lstm_mse = [0.0081, 0.013924, 0.000361, 0.022201, 0.000625, 0.159201, 0.042025, 0.075076, 0.000036, 0.016129,
                  0.200704, 0.173056, 0.042436, 0.039204, 0.299209, 0.0169, 0.093636, 0.034969, 0.055696, 0.118336]
paris_benchmark = ["SP", "LU", "MG", "Myocyte", "Lud", "CG", "IS", "DC", "PuReMD", "Bfs-rodinia", "Kmeans", "Nw", "Backprop", "Blackscholes", "BT", "Hotspot", "Pathfinder", "Nn", "STREAM", "Lulesh"]
paris_mse =   [0.2088, 0.0912, 0.1302, 0.0049, 0.0236, 0.0010, 0.0422, 0.0273, 0.0000, 0.0028, 0.0153, 0.0034, 0.0278, 0.0001, 0.2369, 0.5635, 0.0070, 0.0032, 0.0084, 0.0175]
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

rects1 = ax.bar(index, sorted_lstm_mse, bar_width, alpha=opacity, color='red', label='LSTM')
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
plt.savefig("bar_crash_LSTMvsPARIS.png")