import matplotlib.pyplot as plt

# 与えられた数値リストと対応するノード数
nodes = [20, 30, 40, 50, 60, 70, 80, 90]
times = [
    3.26236415, 9.42217278, 15.18681598, 24.30926299, 
    31.68406987, 75.96639729, 106.47728515, 148.52063799
]

# 平均値を計算
mean_time = sum(times) / len(times)
print("平均値:", mean_time)

# グラフを作成
plt.figure(figsize=(8, 6))
plt.plot(nodes, times, marker='o', linestyle='-', color='b', label='Computation Time')

# グラフの設定
plt.title("Computation Time vs Number of Nodes")
plt.xlabel("Number of Nodes")
plt.ylabel("Computation Time (s)")
plt.axhline(y=mean_time, color='r', linestyle='--', label=f'Average Time: {mean_time:.2f}s')
plt.legend()
plt.grid(True)

# グラフを表示
plt.show()
