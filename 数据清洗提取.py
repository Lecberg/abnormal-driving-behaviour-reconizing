import pandas as pd
from datetime import datetime

# 读取数据
data = pd.read_csv('gps_1101.csv', encoding='utf-8-sig')

# 步骤1：尝试转换时间，处理格式不一致的情况
# 使用 errors='coerce' 将无效时间转换为 NaT
data['gps时间'] = pd.to_datetime(data['gps时间'], format='mixed', errors='coerce')

# 检查并打印无效时间记录的数量
invalid_time_count = data['gps时间'].isna().sum()
print(f"无效时间记录数量: {invalid_time_count}")

# 剔除无效时间记录
data = data.dropna(subset=['gps时间'])

# 步骤2：处理重复采样点，保留GPS速度不为0的点
def handle_duplicates(group):
    # 按时间排序
    group = group.sort_values(by='gps时间')
    # 按时间分组，处理重复时间点
    group = group.groupby('gps时间').apply(lambda x: x[x['gps速度'] != 0].head(1) if (x['gps速度'] != 0).any() else x.head(1))
    return group.reset_index(drop=True)

data = data.groupby('vid_md5').apply(handle_duplicates).reset_index(drop=True)

# 步骤3：剔除GPS速度为0的采样点
data = data[data['gps速度'] != 0]

# 步骤4：提取每辆车采样间隔为1分钟且连续采样不少于10个轨迹点的轨迹
def extract_continuous_trajectories(group):
    if len(group) == 0:
        return pd.DataFrame()
    
    # 按时间排序
    group = group.sort_values(by='gps时间')
    # 计算时间间隔（单位：秒）
    group['time_diff'] = group['gps时间'].diff().dt.total_seconds()
    # 初始化轨迹段标识
    group['segment'] = (group['time_diff'] != 60).cumsum()
    # 按段分组，计算每段的点数
    segments = group.groupby('segment').agg({
        'time_diff': 'count',
        'vid_md5': 'first',
        'gps时间': ['first', 'last']
    }).reset_index()
    segments.columns = ['segment', 'point_count', 'vid_md5', 'start_time', 'end_time']
    # 筛选点数不少于10的段
    valid_segments = segments[segments['point_count'] >= 10]
    if len(valid_segments) == 0:
        return pd.DataFrame()
    # 选择点数最多的段
    max_segment = valid_segments.loc[valid_segments['point_count'].idxmax(), 'segment']
    return group[group['segment'] == max_segment]

# 按车辆分组，提取符合条件的轨迹
result = data.groupby('vid_md5').apply(extract_continuous_trajectories).reset_index(drop=True)

# 保存结果到新文件
result.to_csv('gps_cleared.csv', index=False, encoding='utf-8-sig')

# 输出结果的基本信息
print("清洗后的数据条目数:", len(result))
print("涉及的车辆数:", result['vid_md5'].nunique())
