import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='feature_calculation.log'
)

# 数据验证函数
def validate_gps_coordinates(df):
    """验证GPS坐标的有效性"""
    valid_lat = (df['Lat'] >= -90) & (df['Lat'] <= 90)
    valid_lng = (df['Lng'] >= -180) & (df['Lng'] <= 180)
    invalid_coords = ~(valid_lat & valid_lng)
    if invalid_coords.any():
        logging.warning(f"发现 {invalid_coords.sum()} 条无效GPS坐标记录")
    return df[valid_lat & valid_lng]

def validate_speed(df):
    """验证速度值的合理范围 (0-200 km/h)"""
    valid_speed = (df['gps速度'] >= 0) & (df['gps速度'] <= 200)
    invalid_speed = ~valid_speed
    if invalid_speed.any():
        logging.warning(f"发现 {invalid_speed.sum()} 条异常速度记录")
    df.loc[invalid_speed, 'gps速度'] = np.nan
    return df

# 读取原始数据
logging.info("开始读取数据...")
df = pd.read_csv('gps_cleared.csv', encoding='utf-8-sig')
logging.info(f"共读取 {len(df)} 条记录")

# 将gps时间转换为datetime对象
df['gps时间'] = pd.to_datetime(df['gps时间'])

# 按车辆分组并排序，确保重置索引
df = df.groupby('vid_md5', group_keys=False).apply(lambda x: x.sort_values('gps时间')).reset_index(drop=True)
logging.info(f"处理车辆数量: {df['vid_md5'].nunique()}")

# 计算时间差（单位：秒）
df['时间差'] = df.groupby('vid_md5')['gps时间'].diff().dt.total_seconds()

# 计算加速度
# 1. 首先将速度从km/h转换为m/s (1 km/h = 1000/3600 m/s = 0.277778 m/s)
df['速度_mps'] = df['gps速度'] * 0.277778

# 2. 计算加速度（m/s²）
df['加速度'] = df.groupby('vid_md5')['速度_mps'].diff() / df['时间差'] * 10

# 计算曲折度（以10分钟窗口为例）
window_size = 10 * 60  # 10分钟，单位：秒

def calculate_time_windows(group):
    """为每个车辆单独计算时间窗口"""
    group['时间窗口'] = (group['gps时间'] - group['gps时间'].iloc[0]).dt.total_seconds() // window_size
    return group

df = df.groupby('vid_md5').apply(calculate_time_windows).reset_index(drop=True)

def calculate_distances(group):
    """计算每个窗口的实际距离和直线距离"""
    if len(group) < 2:
        return pd.Series({'实际距离': 0, '直线距离': 0, '曲折度': 1})
    
    # 使用向量化操作计算相邻点之间的距离
    coords = list(zip(group['Lat'], group['Lng']))
    distances = [geodesic(coords[i], coords[i+1]).kilometers for i in range(len(coords)-1)]
    actual_distance = sum(distances)
    
    # 计算起点到终点的直线距离
    straight_distance = geodesic(coords[0], coords[-1]).kilometers
    
    # 避免除以零
    sinuosity = actual_distance / straight_distance if straight_distance > 0 else 1
    return pd.Series({'实际距离': actual_distance, '直线距离': straight_distance, '曲折度': sinuosity})

# 按车辆和时间窗口分组计算曲折度
logging.info("开始计算曲折度...")
sinuosity_df = df.groupby(['vid_md5', '时间窗口'], as_index=False).apply(calculate_distances)
df = df.merge(sinuosity_df[['vid_md5', '时间窗口', '曲折度']], on=['vid_md5', '时间窗口'], how='left')

# 处理缺失值
df[['加速度', '曲折度']] = df[['加速度', '曲折度']].fillna(0)

# 删除临时列
df = df.drop(['时间差', '速度_mps', '时间窗口'], axis=1)

# 输出统计信息
logging.info("计算特征统计信息...")
stats = df[['加速度', '曲折度']].describe()
logging.info("\n" + str(stats))

# 保存结果到CSV（只保存原始列加上新计算的加速度和曲折度）
df.to_csv('gps_features.csv', index=False, encoding='utf-8-sig')
logging.info("特征数据已保存到 gps_features.csv 文件中")

# 输出结果预览
print("\n特征计算结果预览：")
print(df[['加速度', '曲折度']].head())
print("\n特征数据已保存到 gps_features.csv 文件中")