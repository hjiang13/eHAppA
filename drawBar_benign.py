import matplotlib.pyplot as plt

# Data for both benchmarks
lstm_benchmark = ["IS", "LU", "Nn", "Myocyte", "Backprop", "CG", "MG", "BT", "SP", "DC", "Lud", "Kmeans", "STREAM", "Blackscholes", "PuReMD", "Lulesh", "Hotspot", "Bfs-rodinia", "Nw", "Pathfinder"]
lstm_mse = [0.009604, 0.001444, 0.032761, 0.081225, 0.031329, 0.073984, 0.113569, 0.0064, 0.2209, 0.0004, 0.105625, 0.251001, 0.048841, 0.013225, 0.281961, 0.003844, 0.042436, 0.008281, 0.015129, 0.0016]
paris_benchmark = ["SP", "LU", "MG", "Myocyte", "Lud", "CG", "IS", "DC", "PuReMD", "Bfs-rodinia", "Kmeans", "Nw", "Backprop", "Blackscholes", "BT", "Hotspot", "Pathfinder", "Nn", "STREAM", "Lulesh"]
paris_mse = [0.2071, 0.0304, 0.1503, 0.0197, 0.0749, 0.0207, 0.0020, 0.0220, 0.1697, 0.0003, 0.1122, 0.0411, 0.1941, 0.1023, 0.0480, 0.0111, 0.1036, 0.4476, 0.1174, 0.3318]

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

rects1 = ax.bar(index, sorted_lstm_mse, bar_width, alpha=opacity, color='green', label='LSTM')
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
plt.savefig("bar_benign_LSTMvsPARIS.png")