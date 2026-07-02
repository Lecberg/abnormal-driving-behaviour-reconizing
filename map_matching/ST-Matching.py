import networkx as nx
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from rtree import index
import geopandas as gpd
from tqdm import tqdm
from numba import jit
import heapq
from joblib import Parallel, delayed

# ---------------------------
# 1. 加载路网数据并预处理
# ---------------------------
print("Loading road network...")
G = nx.read_graphml(r'C:\Users\Mth13\Desktop\project_code\map_matching\xi_an.graphml')

# 修正节点坐标类型转换
nodes = {node: (float(data['x']), float(data['y'])) for node, data in G.nodes(data=True)}

edges = []
for u, v, data in G.edges(data=True):
    line = LineString([Point(nodes[u]), Point(nodes[v])])
    length = float(data.get('length', line.length))
    edges.append({
        "u": u,
        "v": v,
        "geometry": line,
        "length": length,
        "oneway": data.get('oneway', 'False')
    })

road_gdf = gpd.GeoDataFrame(edges, geometry="geometry")

# ---------------------------
# 2. 构建有向路网图
# ---------------------------
print("Building directed road graph...")
road_graph = nx.DiGraph()
for _, road in road_gdf.iterrows():
    road_graph.add_edge(road['u'], road['v'], weight=road['length'])
    if road['oneway'].lower() != 'true':
        road_graph.add_edge(road['v'], road['u'], weight=road['length'])

# ---------------------------
# 3. 加载GPS轨迹数据
# ---------------------------
print("Loading GPS data...")
gps_df = pd.read_csv(r"C:\Users\Mth13\Desktop\project_code\map_matching\gps_features.csv")
gps_points = [Point(row['Lng'], row['Lat']) for _, row in gps_df.iterrows()]

# ---------------------------
# 4. 生成候选点并预计算路径
# ---------------------------
MAX_CANDIDATES = 5
SIGMA = 50.0
BEAM_WIDTH = 50
BETA = 50.0

@jit(nopython=True)
def compute_trans_prob(path_length, euclidean_dist, beta):
    return np.exp(-abs(path_length - euclidean_dist) / beta) / beta

# 生成候选点
print("Generating candidates...")
idx = index.Index()
for i, road in road_gdf.iterrows():
    idx.insert(i, road.geometry.bounds)

candidates = []
all_nodes = set()

for t, gps_point in enumerate(tqdm(gps_points)):
    nearby_road_ids = list(idx.nearest((gps_point.x, gps_point.y), num_results=MAX_CANDIDATES))
    current_candidates = []
    for road_id in nearby_road_ids:
        road = road_gdf.iloc[road_id]
        line = road.geometry
        projected_point = line.interpolate(line.project(gps_point))
        distance = gps_point.distance(projected_point)
        obs_prob = np.exp(-0.5 * (distance / SIGMA)**2) / (SIGMA * np.sqrt(2 * np.pi))
        current_candidates.append({
            "road_id": road_id,
            "point": projected_point,
            "obs_prob": obs_prob,
            "u": road['u'],
            "v": road['v']
        })
        all_nodes.update([road['u'], road['v']])
    candidates.append(current_candidates)

# 预计算关键节点间的最短路径（并行化优化）
print("Precomputing shortest paths with parallel processing...")
all_nodes = list(all_nodes)

def compute_single_source_distances(u, graph, target_nodes):
    try:
        lengths = nx.single_source_dijkstra_path_length(graph, u, weight='weight')
        return {u: {v: lengths.get(v, np.inf) for v in target_nodes}}
    except nx.NodeNotFound:
        return {u: {v: np.inf for v in target_nodes}}

# 并行计算最短路径
results = Parallel(n_jobs=-1)(
    delayed(compute_single_source_distances)(u, road_graph, all_nodes)
    for u in tqdm(all_nodes, desc="Distributing tasks")
)

# 合并结果到距离矩阵
distance_matrix = {}
for result in results:
    distance_matrix.update(result)

# ---------------------------
# 5. 加速版动态规划
# ---------------------------
print("Running optimized Viterbi...")
dp = [{} for _ in range(len(gps_points))]
backpointers = [{} for _ in range(len(gps_points))]

# 初始化第一个时间步
if len(candidates[0]) > 0:
    top_k = heapq.nlargest(BEAM_WIDTH, candidates[0], key=lambda x: x['obs_prob'])
    for cand in top_k:
        dp[0][cand['road_id']] = cand['obs_prob']
        backpointers[0][cand['road_id']] = None

# 时间步迭代
for t in tqdm(range(1, len(gps_points)), desc="Processing"):
    if len(candidates[t]) == 0 or len(dp[t-1]) == 0:
        continue
    
    # 获取前一时间步的Top-K候选
    prev_top = heapq.nlargest(BEAM_WIDTH, dp[t-1].items(), key=lambda x: x[1])
    
    for curr_cand in candidates[t]:
        max_prob = -np.inf
        best_prev = None
        
        for (prev_road_id, prev_prob) in prev_top:
            prev_cand = next(c for c in candidates[t-1] if c['road_id'] == prev_road_id)
            
            # 从预计算矩阵获取路径长度
            try:
                path_length = distance_matrix[prev_cand['v']][curr_cand['u']]
            except KeyError:
                path_length = np.inf
            
            # 计算转移概率
            euclidean_dist = prev_cand['point'].distance(curr_cand['point'])
            trans_prob = compute_trans_prob(path_length, euclidean_dist, BETA)
            
            total_prob = prev_prob * trans_prob * curr_cand['obs_prob']
            if total_prob > max_prob:
                max_prob = total_prob
                best_prev = prev_road_id
        
        if max_prob > 0:
            dp[t][curr_cand['road_id']] = max_prob
            backpointers[t][curr_cand['road_id']] = best_prev
    
    # 剪枝保留Top-K
    dp[t] = dict(heapq.nlargest(BEAM_WIDTH, dp[t].items(), key=lambda x: x[1]))

# ---------------------------
# 6. 回溯路径
# ---------------------------
print("Backtracking path...")
best_path = []
if len(dp[-1]) > 0:
    current_road = max(dp[-1], key=lambda k: dp[-1][k])
    best_path.append(current_road)
    for t in range(len(gps_points)-1, 0, -1):
        current_road = backpointers[t].get(current_road)
        if current_road is None:
            break
        best_path.insert(0, current_road)

# ---------------------------
# 7. 输出结果
# ---------------------------
print("Saving results...")
# 创建与GPS点数量匹配的highway列表（初始为None）
matched_highway = [None] * len(gps_points)
# 填充匹配结果
for t, road_id in enumerate(best_path):
    if road_id is not None:
        try:
            matched_highway[t] = road_gdf.iloc[road_id].get('highway', 'unknown')
        except IndexError:
            matched_highway[t] = 'invalid_road_id'
# 将匹配结果添加到原始GPS数据
gps_df['matched_highway'] = matched_highway
# 保存增强后的GPS数据
output_path = r"C:\Users\Mth13\Desktop\project_code\map_matching\gps_features_with_highway.csv"
gps_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Matching completed! Enhanced data saved to {output_path}")
