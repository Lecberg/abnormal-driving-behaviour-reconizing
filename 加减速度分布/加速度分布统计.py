import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['DengXian', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'PingFang SC', 'Hiragino Sans GB', 'SimHei']  # 添加更多现代字体
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.size'] = 12  # 设置字体大小

# 读取数据
print("正在读取数据...")
df = pd.read_csv(r'C:\Users\Mth13\Desktop\毕设code\gps_features.csv', encoding='utf-8-sig')

# 分离加速度和减速度
acceleration = df[df['加速度'] > 0]['加速度']
deceleration = df[df['加速度'] < 0]['加速度'].abs()  # 取绝对值便于比较

print("\n数据概况：")
print(f"加速度样本数: {len(acceleration)}")
print(f"减速度样本数: {len(deceleration)}")

# 基本统计量
print("\n加速度（正值）基本统计量：")
acc_stats = acceleration.describe()
print(acc_stats)

print("\n减速度（正值）基本统计量：")
dec_stats = deceleration.describe()
print(dec_stats)

# 计算分位数
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
acc_percentiles = np.percentile(acceleration.dropna(), percentiles)
dec_percentiles = np.percentile(deceleration.dropna(), percentiles)

print("\n加速度分位数：")
for p, v in zip(percentiles, acc_percentiles):
    print(f"{p}% 分位数: {v:.2f} m/s²")

print("\n减速度分位数：")
for p, v in zip(percentiles, dec_percentiles):
    print(f"{p}% 分位数: {v:.2f} m/s²")

# 创建加速度图形
plt.figure(figsize=(12, 5))

# 1. 加速度直方图
plt.subplot(1, 2, 1)
sns.histplot(data=acceleration, bins=50, kde=True)
plt.title('加速度分布直方图')
plt.xlabel('加速度 (m/s²)', fontname='DengXian')  # 直接使用²字符，并指定字体
plt.ylabel('频数')

# 2. 加速度箱线图
plt.subplot(1, 2, 2)
sns.boxplot(y=acceleration)
plt.title('加速度箱线图')
plt.ylabel('加速度 (m/s²)', fontname='DengXian')  # 直接使用²字符，并指定字体

# 调整布局
plt.tight_layout()

# 保存加速度图形
plt.savefig('加减速度分布\加速度分布分析.png', dpi=300, bbox_inches='tight')
print("\n加速度分析图表已保存为 '加速度分布分析.png'")

# 创建减速度图形
plt.figure(figsize=(12, 5))

# 1. 减速度直方图
plt.subplot(1, 2, 1)
sns.histplot(data=deceleration, bins=50, kde=True)
plt.title('减速度分布直方图')
plt.xlabel('减速度 (m/s²)', fontname='DengXian')  # 直接使用²字符，并指定字体
plt.ylabel('频数')

# 2. 减速度箱线图
plt.subplot(1, 2, 2)
sns.boxplot(y=deceleration)
plt.title('减速度箱线图')
plt.ylabel('减速度 (m/s²)', fontname='DengXian')  # 直接使用²字符，并指定字体

# 调整布局
plt.tight_layout()

# 保存减速度图形
plt.savefig('加减速度分布\减速度分布分析.png', dpi=300, bbox_inches='tight')
print("减速度分析图表已保存为 '减速度分布分析.png'")

# 计算统计特征
acc_skewness = acceleration.skew()
acc_kurtosis = acceleration.kurtosis()
dec_skewness = deceleration.skew()
dec_kurtosis = deceleration.kurtosis()

print(f"\n分布形态特征：")
print("加速度：")
print(f"偏度: {acc_skewness:.2f}")
print(f"峰度: {acc_kurtosis:.2f}")
print("\n减速度：")
print(f"偏度: {dec_skewness:.2f}")
print(f"峰度: {dec_kurtosis:.2f}")

# 正态性检验
acc_stat, acc_p = stats.normaltest(acceleration.dropna())
dec_stat, dec_p = stats.normaltest(deceleration.dropna())

print(f"\n正态性检验：")
print("加速度：")
print(f"统计量: {acc_stat:.2f}")
print(f"P值: {acc_p:.4f}")
print("减速度：")
print(f"统计量: {dec_stat:.2f}")
print(f"P值: {dec_p:.4f}")

# 将统计结果保存到文件
with open('加减速度分布\加减速度统计结果.txt', 'w', encoding='utf-8') as f:
    f.write("加减速度分布统计分析报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("0. 数据概况\n")
    f.write("-" * 30 + "\n")
    f.write(f"加速度样本数: {len(acceleration)}\n")
    f.write(f"减速度样本数: {len(deceleration)}\n\n")
    
    f.write("1. 基本统计量\n")
    f.write("-" * 30 + "\n")
    f.write("加速度（正值）：\n")
    f.write(str(acc_stats) + "\n\n")
    f.write("减速度（正值）：\n")
    f.write(str(dec_stats) + "\n\n")
    
    f.write("2. 分位数统计\n")
    f.write("-" * 30 + "\n")
    f.write("加速度：\n")
    for p, v in zip(percentiles, acc_percentiles):
        f.write(f"{p}% 分位数: {v:.2f} m/s²\n")
    f.write("\n减速度：\n")
    for p, v in zip(percentiles, dec_percentiles):
        f.write(f"{p}% 分位数: {v:.2f} m/s²\n")
    f.write("\n")
    
    f.write("3. 分布形态特征\n")
    f.write("-" * 30 + "\n")
    f.write("加速度：\n")
    f.write(f"偏度: {acc_skewness:.2f}\n")
    f.write(f"峰度: {acc_kurtosis:.2f}\n\n")
    f.write("减速度：\n")
    f.write(f"偏度: {dec_skewness:.2f}\n")
    f.write(f"峰度: {dec_kurtosis:.2f}\n\n")
    
    f.write("4. 正态性检验\n")
    f.write("-" * 30 + "\n")
    f.write("加速度：\n")
    f.write(f"统计量: {acc_stat:.2f}\n")
    f.write(f"P值: {acc_p:.4f}\n")
    if acc_p < 0.05:
        f.write("结论：数据分布显著偏离正态分布\n\n")
    else:
        f.write("结论：数据分布近似服从正态分布\n\n")
    
    f.write("减速度：\n")
    f.write(f"统计量: {dec_stat:.2f}\n")
    f.write(f"P值: {dec_p:.4f}\n")
    if dec_p < 0.05:
        f.write("结论：数据分布显著偏离正态分布\n")
    else:
        f.write("结论：数据分布近似服从正态分布\n")

print("\n详细的统计结果已保存到 '加减速度统计结果.txt' 文件中") 