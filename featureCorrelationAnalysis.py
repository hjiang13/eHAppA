import pandas as pd

# Creating a DataFrame from the provided data

df = pd.read_csv("keyFeature.csv")

# Splitting keywords into keys by ','
df['Keywords'] = df['Keywords'].apply(lambda x: x.split(','))

# Creating a dictionary to hold the initial score of all keys
key_scores = {}
k  = 10
# SDC group sets
SDC_better_keywords = set(['IS', 'STREAM', 'PuReMD', 'Kmeans', 'Lulesh', "MG", "LU", "Bfs-rodinia", "CG", "NW"])
SDC_worse_keywords = set(['DC', 'Blacksholes', 'Hotspot', 'Lud', "SP", "Nn", "Pathfinder"])

benign_better_keywords = set(["DC","LU","Pathfinder","Lulesh","BT","Bfs-rodinia","IS","Blackholes","Nw","Backprop","Nn","Hotspot","STREAM","Myocyte","Lud"])
benign_worse_keywords = set(["CG","MG","SP","Kmeans","PuReMD"])

crash_better_keywords = set(["SP","Nn","Backprop","IS","Myocyte","Blacksholes","PuReMD"])
crash_worse_keywords = set(["LU","DC","Lulesh","Bfs-rodinia","MG","STREAM","NW","BT","Hotspot","Pathfinder","CG","Kmeans","Lud"])

# Increment and decrement scores based on the SDC groups
for index, row in df.iterrows():
    bench = row['Filename']
    keywords = row['Keywords']
    for key in keywords:
        if key not in key_scores:
            key_scores[key] = 0
        if bench in SDC_better_keywords:
            key_scores[key] += 1
        if bench in SDC_worse_keywords:
            key_scores[key] -= 1

# Sorting the scores
sorted_scores = sorted(key_scores.items(), key=lambda item: item[1], reverse=True)

# Getting the top 20 and last 20 keys based on scores
top_20 = sorted_scores[:k]
last_20 = sorted_scores[-k:]
print("SDC: ")

print(top_20, last_20)

# Increment and decrement scores based on the benign groups
key_scores = {}
for index, row in df.iterrows():
    bench = row['Filename']
    keywords = row['Keywords']
    for key in keywords:
        if key not in key_scores:
            key_scores[key] = 0
        if bench in benign_better_keywords:
            key_scores[key] += 1
        if bench in benign_worse_keywords:
            key_scores[key] -= 1

# Sorting the scores
sorted_scores = sorted(key_scores.items(), key=lambda item: item[1], reverse=True)

# Getting the top 20 and last 20 keys based on scores
top_20 = sorted_scores[:k]
last_20 = sorted_scores[-k:]
print("benign: ")

print(top_20, last_20)

# Increment and decrement scores based on the crash groups
key_scores = {}
for index, row in df.iterrows():
    bench = row['Filename']
    keywords = row['Keywords']
    for key in keywords:
        if key not in key_scores:
            key_scores[key] = 0
        if bench in crash_better_keywords:
            key_scores[key] += 1
        if bench in crash_worse_keywords:
            key_scores[key] -= 1

# Sorting the scores
sorted_scores = sorted(key_scores.items(), key=lambda item: item[1], reverse=True)

# Getting the top 20 and last 20 keys based on scores
top_20 = sorted_scores[:k]
last_20 = sorted_scores[-k:]
print("crash: ")

print(top_20, last_20)
