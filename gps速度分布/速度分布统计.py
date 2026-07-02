import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import numpy as np

# 设置Seaborn样式
sns.set_style("whitegrid")  # 使用seaborn的网格样式

# 指定中文字体路径并添加到字体管理器
font_path = r"C:/Windows/Fonts/simhei.ttf"  # 如果路径不同，请自行修改
fm.fontManager.addfont(font_path)
font_prop = FontProperties(fname=font_path)
font_name = font_prop.get_name()

# 设置全局字体和符号显示
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('gps_features.csv')

# 创建图形 - 修改为横向布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制直方图
sns.histplot(data=df, x='gps速度', bins=30, ax=ax1)
ax1.set_title('GPS速度分布直方图', fontproperties=font_prop, fontsize=14)
ax1.set_xlabel('速度 (km/h)', fontproperties=font_prop, fontsize=12)
ax1.set_ylabel('频数', fontproperties=font_prop, fontsize=12)
ax1.tick_params(labelsize=10)

# 绘制箱线图
sns.boxplot(y=df['gps速度'], ax=ax2)
ax2.set_title('GPS速度箱线图', fontproperties=font_prop, fontsize=14)
ax2.set_ylabel('速度 (km/h)', fontproperties=font_prop, fontsize=12)
ax2.tick_params(labelsize=10)

# 计算分位数
quartiles = df['gps速度'].quantile([0, 0.25, 0.5, 0.75, 1.0])

# 计算其他统计信息
mean = df['gps速度'].mean()
std = df['gps速度'].std()
count = df['gps速度'].count()

# 将统计结果保存到文件
with open('gps速度分布/速度统计结果.txt', 'w', encoding='utf-8') as f:
    f.write('GPS速度统计结果\n')
    f.write('=' * 20 + '\n')
    f.write(f'最小值: {quartiles[0]:.1f} km/h\n')
    f.write(f'下四分位数: {quartiles[0.25]:.1f} km/h\n')
    f.write(f'中位数: {quartiles[0.5]:.1f} km/h\n')
    f.write(f'上四分位数: {quartiles[0.75]:.1f} km/h\n')
    f.write(f'最大值: {quartiles[1]:.1f} km/h\n')
    f.write(f'均值: {mean:.1f} km/h\n')
    f.write(f'标准差: {std:.1f} km/h\n')
    f.write(f'样本数: {count}\n')

# 在箱线图上添加标注
def add_value_label(value, y_offset=0.02):
    ax2.text(-0.15, value, f'{value:.1f}', 
             horizontalalignment='right',
             verticalalignment='center',
             fontsize=10)

# 添加各个分位数的标注
add_value_label(quartiles[0])      # 最小值
add_value_label(quartiles[0.25])   # 下四分位数
add_value_label(quartiles[0.5])    # 中位数
add_value_label(quartiles[0.75])   # 上四分位数
add_value_label(quartiles[1])      # 最大值

# 添加坐标轴刻度的中文字体显示
for ax in (ax1, ax2):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('gps速度分布/速度分布图.png', dpi=300, bbox_inches='tight')
plt.show()
