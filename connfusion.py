import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
# 给定的混淆矩阵
conf_matrix_another = np.array([
 [1256,   21,   13,   17 , 2,  102,   15,   20,    1],
 [   0,  469 ,  41   , 4  ,  0  ,  8  ,  0 ,   0  ,  0],
 [   0 ,   3 , 520 ,   0  ,  1  ,  0  ,  0  ,  0  ,  0],
 [   0 ,   0  ,  0 , 704 ,  1  , 14  ,  0   , 0   , 0],
 [   0 ,   3  ,  0 ,   3,  649  ,  3 ,   0 ,   0  ,  0],
 [   0 ,   6  ,  0 , 171  ,  1 , 537,    0  ,  0 ,   0],
 [   0  ,  0  ,  0 ,   0  ,  0  ,  0,  524 ,   0    ,0],
 [   0  ,  0 ,   0 ,   0  ,  0   , 0   , 0,  524 ,   0],
 [   0   , 0  ,  1  ,  0  ,  1  ,  0   , 0  ,  0 , 522]
])

# 设置混淆矩阵的标签（英文类名）
class_names = ['Walk', 'Sit', 'Stretch', 'Lie', 'Stand', 'Crawl', 'Jump', 'Wave', 'Bend']

# 交换第二类和第九类，交换第八类和第四类
conf_matrix_another[[1, 8], :] = conf_matrix_another[[8, 1], :]
conf_matrix_another[:, [1, 8]] = conf_matrix_another[:, [8, 1]]
conf_matrix_another[[3, 7], :] = conf_matrix_another[[7, 3], :]
conf_matrix_another[:, [3, 7]] = conf_matrix_another[:, [7, 3]]

# 绘制热力图
plt.figure(figsize=(10, 8))


# 计算每一行的总和
row_sums = np.sum(conf_matrix_another, axis=1)
# 计算每一行的百分比
percent_matrix = conf_matrix_another / row_sums[:, np.newaxis] 


ax=sns.heatmap(percent_matrix, annot=True, fmt=".3f", cmap="Blues",xticklabels=class_names, yticklabels=class_names, cbar=True)
# 设置标签的颜色为白色
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_color('white')

# 获取颜色条对象并设置其刻度数字的颜色为白色
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelcolor='white')

# 添加标题并显示准确率
plt.title(f'Confusion Matrix', fontsize=16,  color='white',pad=20)
plt.xlabel('Predicted Labels', fontsize=12, color='white', labelpad=15)
plt.ylabel('True Labels', fontsize=12,color='white', labelpad=15)

rect = patches.Rectangle(
    (0, 0), 1, 1, transform=ax.transAxes, color='black', linewidth=1, fill=False
)
ax.add_patch(rect)

# 保存图片
plt.savefig('/root/projects/JJW_Model/contrast/png/open.png', dpi=900, bbox_inches='tight')
