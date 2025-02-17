import matplotlib.pyplot as plt

# Data from the table
timepatch = [10, 20, 30, 40, 50, 60, 70, 80]
model = [0.9275, 0.9417, 0.9512, 0.9652, 0.9646, 0.9786, 0.9758, 0.9664]

# Plotting the data
plt.figure(figsize=(8, 5))
plt.plot(timepatch, model, marker='o', markersize=3, color='#00008B', linewidth=1)

# Adding labels and title
plt.title('Performance of Model Across Timepatch Sizes', fontsize=14)
plt.xlabel('Timepatch Size', fontsize=12)
plt.ylabel('Model Accuracy', fontsize=12)
plt.xticks(timepatch, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0.9, 1)
plt.xlim(0, 85)
plt.grid(True, linestyle='--', alpha=0.7)

for x, y in zip(timepatch, model):
    plt.text(x, y, f'{y:.4f}', fontsize=9, ha='left', va='bottom')

# Show the plot
plt.tight_layout()
# 保存图片
plt.savefig('/root/projects/JJW_Model/contrast/png/darw2.png', dpi=900, bbox_inches='tight')
