# gps_map_matching_osmnx.py
import pandas as pd
import geopandas as gpd
from shapely import wkt
from pyproj import Transformer
import matplotlib.pyplot as plt
import random
import time
import os
import leuvenmapmatching # 确保基础库导入

# --- LeuvenMapMatching Imports ---
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap # 使用 InMemMap

# --- OSMnx for loading OSM data ---
import osmnx as ox # 导入 osmnx

# --- Multiprocessing ---
import multiprocessing as mp
from functools import partial

# --- 1. 配置 ---
OSM_PBF_FILE = 'map_matching/shaanxi-latest.osm.pbf' # PBF 文件路径
# 脚本将期望对应的 .osm 文件存在于同一目录
OSM_XML_FILE = OSM_PBF_FILE.replace(".osm.pbf", ".osm") 

GPS_TRAJECTORIES_FILE = 'map_matching/gps_features.csv'

# LeuvenMapMatching 参数
LMM_MAX_DIST = 150.0
LMM_MAX_DIST_INIT = 100.0
LMM_OBS_NOISE = 30.0
LMM_DIST_NOISE = 50.0
LMM_MIN_PROB_NORM = 0.001
# LMM_MAX_LATTICE_WIDTH = 10 # 可选

VISUALIZATION_TARGET_CRS = "EPSG:32649" # 可视化时投影到的目标坐标系


# --- 2. 使用 OSMnx 和 InMemMap 加载路网数据 ---
def load_map_with_osmnx_and_inmem(osm_xml_filepath, network_type="drive"):
    print(f"使用 osmnx 从 OSM XML 文件加载路网: {osm_xml_filepath}")
    if not os.path.exists(osm_xml_filepath):
        print(f"错误: 找不到 OSM XML 文件: {osm_xml_filepath}")
        print(f"请确保已将 {OSM_PBF_FILE} 转换为 {osm_xml_filepath} (例如使用 osmconvert).")
        return None
    
    try:
        # 1. 使用 osmnx 从 .osm XML 文件加载路网图
        #    osmnx 的图节点是 OSM ID，边包含几何和其他属性。坐标是经纬度。
        print(f"osmnx: 正在从 {osm_xml_filepath} 构建图 (network_type='{network_type}')... 这可能需要一些时间。")
        start_ox = time.time()
        # ox.settings.log_console=True # 可以打开osmnx的日志输出
        # ox.settings.use_cache=True # 启用osmnx的缓存
        
        # 'drive' 通常用于获取适合车辆行驶的道路
        # retain_all=True 确保图是连通的（如果可能）
        # simplify=True 会简化图的拓扑，移除不必要的节点，推荐用于路由
        graph_ox = ox.graph_from_xml(osm_xml_filepath, network_type=network_type, simplify=True, retain_all=True)
        print(f"osmnx: 图构建完成，耗时 {time.time() - start_ox:.2f} 秒。节点数: {len(graph_ox.nodes)}, 边数: {len(graph_ox.edges)}")

        # 2. 使用 InMemMap 包装 osmnx 图
        #    InMemMap(use_latlon=True) 会期望图中的节点有 'x' (lon) 和 'y' (lat) 属性，
        #    并会在内部将图投影到局部UTM带进行匹配。
        #    它还会构建R-tree索引 (如果 use_rtree=True, 默认是False，需要显式开启)。
        print("InMemMap: 正在使用 osmnx 图初始化 (use_latlon=True, use_rtree=True)...")
        start_inmem = time.time()
        # InMemMap 会自动从 osmnx 图中提取必要的边属性，如 'length', 'id' (它会用 u,v,key 作为id)
        # 确保 osmnx 图的边有 'length' 属性 (ox.add_edge_speeds 和 ox.add_edge_travel_times 后通常会有)
        # 如果没有，InMemMap 可能会尝试自己计算。
        # 为了确保有 'length'，可以预先计算：
        graph_ox = ox.add_edge_bearings(graph_ox) # bearings 可能对某些分析有用
        # graph_ox = ox.add_edge_speeds(graph_ox) # 如果需要速度和时间
        # graph_ox = ox.add_edge_travel_times(graph_ox)
        # ox.utils_graph.graph_to_gdfs(graph_ox, nodes=False)["length"] 可以获取边长度
        # InMemMap 应该能处理没有预计算长度的情况，但预计算可能更稳妥。
        # 对于 InMemMap，它期望边上有 'length' 属性。osmnx 的 simplify=True 后的图的边通常有 'length'。

        map_con = InMemMap(name="shaanxi_osmnx", graph=graph_ox, use_latlon=True, use_rtree=True,เส้นทางแคชไดเรกทอรี="map_cache_osmnx")
        # dir_cache="map_cache_osmnx" # InMemMap 也有缓存机制
        print(f"InMemMap: 初始化完成，耗时 {time.time() - start_inmem:.2f} 秒。")
        
        print("OSMnx 路网加载并包装为 InMemMap 完成。")
        return map_con

    except Exception as e:
        print(f"错误: 使用 osmnx 和 InMemMap 加载路网失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 3. 加载GPS轨迹数据 (与之前版本相同，返回 (lat,lon) 列表) ---
def load_gps_trajectories(filepath):
    print(f"加载GPS轨迹数据从 {filepath}...")
    try:
        df_gps = pd.read_csv(filepath)
    except Exception as e:
        print(f"错误：加载CSV文件失败: {e}")
        return {}

    required_cols = ['vid_md5', 'Lng', 'Lat']
    if not all(col in df_gps.columns for col in required_cols):
        print(f"错误：CSV文件必须包含列: {required_cols}")
        return {}

    time_col_found = False
    for col_name in ['标准时间', 'timestamp', 'time']:
        if col_name in df_gps.columns:
            try:
                print(f"按时间列排序: {col_name}")
                df_gps[col_name] = pd.to_datetime(df_gps[col_name])
                df_gps.sort_values(by=['vid_md5', col_name], inplace=True)
                time_col_found = True
                break
            except Exception as e:
                print(f"警告：无法解析或排序时间列 {col_name}: {e}")
    if not time_col_found:
        print("警告：未找到可识别的时间列用于排序轨迹。")
    
    trajectories_latlon = {}
    for vehicle_id, group in df_gps.groupby('vid_md5'):
        trace_latlon_points = []
        for _, row in group.iterrows():
            trace_latlon_points.append((row['Lat'], row['Lng'])) # (latitude, longitude)
        
        if trace_latlon_points:
            trajectories_latlon[vehicle_id] = trace_latlon_points
            
    print(f"加载了 {len(trajectories_latlon)} 辆车的GPS轨迹。")
    return trajectories_latlon


# --- Worker function for multiprocessing (与之前版本相同) ---
def match_vehicle_leuven_worker(vehicle_data_tuple, map_con_ref, matcher_params_ref):
    vehicle_id, trace_latlon = vehicle_data_tuple
    
    if not trace_latlon or len(trace_latlon) < 2:
        return vehicle_id, {"original_trace_latlon": trace_latlon, "matched_path_wkt": None, "states": None, "error": "TRACE_TOO_SHORT"}

    try:
        matcher = DistanceMatcher(
            map_con_ref, # InMemMap 实例
            max_dist=matcher_params_ref['max_dist'],
            max_dist_init=matcher_params_ref['max_dist_init'],
            obs_noise=matcher_params_ref['obs_noise'],
            dist_noise=matcher_params_ref['dist_noise'],
            min_prob_norm=matcher_params_ref['min_prob_norm'],
            max_lattice_width=matcher_params_ref.get('max_lattice_width', 20),
            non_emitting_states=True,
            non_emitting_length_factor=2.0,
        )
        
        status, lmm_states = matcher.match_trace(trace_latlon) # 传递 (lat,lon) 轨迹
        
        matched_path_wkt = None
        if status == matcher.STATUS_OK or status == matcher.STATUS_MATCHED_PARTIALLY:
            matched_path_wkt = matcher.lattice_best_path_to_wkt()
        else:
            return vehicle_id, {"original_trace_latlon": trace_latlon, "matched_path_wkt": None, "states": lmm_states, "error": f"MATCH_STATUS_{status}"}

        return vehicle_id, {"original_trace_latlon": trace_latlon, "matched_path_wkt": matched_path_wkt, "states": lmm_states}

    except Exception as e:
        # print(f"错误：车辆 {vehicle_id} 在 LeuvenMapMatching 匹配过程中发生异常: {e}")
        return vehicle_id, {"original_trace_latlon": trace_latlon, "matched_path_wkt": None, "states": None, "error": str(e)}


# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    overall_start_time = time.time()

    print("--- 步骤1: 使用OSMnx和InMemMap加载路网数据 ---")
    load_roads_start_time = time.time()
    map_con = load_map_with_osmnx_and_inmem(OSM_XML_FILE) # 加载转换后的 .osm 文件
    print(f"路网加载和InMemMap初始化耗时 {time.time() - load_roads_start_time:.2f} 秒")

    if not map_con or not map_con.graph: # InMemMap 有 graph 属性
        print("\n错误：未能成功加载路网数据。程序退出。")
        exit()

    print("\n--- 步骤2: 加载GPS轨迹 ---")
    load_gps_start_time = time.time()
    all_gps_trajectories_latlon = load_gps_trajectories(GPS_TRAJECTORIES_FILE)
    print(f"GPS数据加载耗时 {time.time() - load_gps_start_time:.2f} 秒")

    if not all_gps_trajectories_latlon:
        print("\n错误：未能成功加载GPS轨迹数据。程序退出。")
        exit()

    all_vehicle_ids = list(all_gps_trajectories_latlon.keys())
    if not all_vehicle_ids:
        print("\n没有GPS轨迹需要处理。")
        exit()
        
    print(f"\n--- 步骤3: 使用LeuvenMapMatching进行路网匹配 (共 {len(all_vehicle_ids)} 辆车) ---")
    matching_start_time = time.time()
    
    lmm_params = {
        'max_dist': LMM_MAX_DIST,
        'max_dist_init': LMM_MAX_DIST_INIT,
        'obs_noise': LMM_OBS_NOISE,
        'dist_noise': LMM_DIST_NOISE,
        'min_prob_norm': LMM_MIN_PROB_NORM,
    }
    
    tasks = [(vid, all_gps_trajectories_latlon[vid]) for vid in all_vehicle_ids if all_gps_trajectories_latlon.get(vid)]

    partial_lmm_worker = partial(match_vehicle_leuven_worker,
                                 map_con_ref=map_con,
                                 matcher_params_ref=lmm_params)

    num_processes = max(1, mp.cpu_count() - 2) # 留更多核心给系统，特别是osmnx可能也耗CPU
    if len(tasks) < num_processes * 2 : # 如果任务太少，减少进程数
        num_processes = max(1, len(tasks) // 2)
    if num_processes == 0 and len(tasks) > 0:
        num_processes = 1

    # num_processes = 1 # DEBUG: 单进程测试
    print(f"使用 {num_processes} 个进程进行匹配。")

    lmm_results_collected = {}
    if num_processes > 0:
        with mp.Pool(processes=num_processes) as pool:
            processed_lmm_results_list = pool.map(partial_lmm_worker, tasks)
        for vehicle_id, match_output_dict in processed_lmm_results_list:
            lmm_results_collected[vehicle_id] = match_output_dict
    else: # 如果没有任务或无法创建进程
        print("没有任务或无法创建进程，跳过并行处理。")
            
    print(f"LeuvenMapMatching 完成了 {len(lmm_results_collected)} 辆车的匹配。")
    print(f"总匹配耗时: {time.time() - matching_start_time:.2f} 秒")

    # --- 步骤4: 可视化 (与之前版本相同) ---
    print("\n--- 步骤4: 可视化选定车辆的匹配结果 ---")
    visualization_start_time = time.time()

    if not lmm_results_collected:
        print("没有匹配结果可供可视化。")
        exit()

    num_vehicles_to_plot = min(5, len(lmm_results_collected))
    
    available_vehicle_ids_for_plot = [
        vid for vid, res_data in lmm_results_collected.items()
        if res_data and res_data.get("matched_path_wkt")
    ]

    if not available_vehicle_ids_for_plot:
        print("没有车辆具有有效的匹配路径可供可视化。")
        exit()
        
    vehicle_ids_to_plot = random.sample(available_vehicle_ids_for_plot, 
                                        min(num_vehicles_to_plot, len(available_vehicle_ids_for_plot)))
    
    print(f"将为以下车辆生成独立绘图: {vehicle_ids_to_plot}")

    gps_transformer_for_viz = Transformer.from_crs("EPSG:4326", VISUALIZATION_TARGET_CRS, always_xy=True)

    # 可视化路网背景：从 InMemMap.graph 获取边并投影
    # InMemMap(use_latlon=True) 内部会将图投影到局部UTM。
    # 我们需要获取这些投影后的几何，或者重新投影原始OSM图用于可视化。
    # 为了与匹配路径WKT（已经是米制）一致，我们需要米制的路网。
    # InMemMap.graph 中的 'geom' 属性可能已经是投影后的 LineString。
    base_road_geoms_for_plot = []
    if map_con and map_con.graph:
        # 假设 InMemMap 已经将几何投影到了某个米制系统，并且存储在 'geom' 属性中
        # 并且这个米制系统与 lattice_best_path_to_wkt() 返回的WKT的坐标系一致
        # (通常 InMemMap 会选择一个合适的局部UTM带)
        # 如果不一致，可视化会出问题。
        # 一个更稳妥的方法是，如果知道 InMemMap 内部使用的投影，就用那个。
        # 或者，重新从 osmnx_graph 投影到 VISUALIZATION_TARGET_CRS。
        
        # 尝试直接使用 InMemMap.graph 中的 'geom'
        # 注意：InMemMap 的边 key 可能不是简单的 0。
        # 它可能使用 (u,v,osmid) 或类似作为边标识。
        # 我们需要遍历所有边。
        for u, v, data in map_con.graph.edges(data=True):
            if 'geom' in data and data['geom'] and not data['geom'].is_empty:
                # 假设这个 geom 已经是 InMemMap 内部使用的米制坐标
                base_road_geoms_for_plot.append(data['geom'])
            # else: # 如果没有 'geom'，可能需要从节点坐标构建
            #     if 'x' in map_con.graph.nodes[u] and 'y' in map_con.graph.nodes[u] and \
            #        'x' in map_con.graph.nodes[v] and 'y' in map_con.graph.nodes[v]:
            #         p1 = (map_con.graph.nodes[u]['x'], map_con.graph.nodes[u]['y'])
            #         p2 = (map_con.graph.nodes[v]['x'], map_con.graph.nodes[v]['y'])
            #         base_road_geoms_for_plot.append(LineString([p1, p2]))


    gs_roads_plot = gpd.GeoSeries(list(set(base_road_geoms_for_plot))) if base_road_geoms_for_plot else None # 使用set去重


    for i, vehicle_id in enumerate(vehicle_ids_to_plot):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        result_data = lmm_results_collected[vehicle_id]
        original_trace_latlon = result_data.get("original_trace_latlon") 
        matched_path_wkt_str = result_data.get("matched_path_wkt")

        # 绘制路网背景 (米制)
        if gs_roads_plot is not None and not gs_roads_plot.empty:
            gs_roads_plot.plot(ax=ax, linewidth=0.5, color='lightgray', alpha=0.8, zorder=1)


        plot_title = f"LMM (OSMnx+InMemMap): 车辆 {vehicle_id}"
        min_x_viz, max_x_viz, min_y_viz, max_y_viz = None, None, None, None

        if original_trace_latlon and isinstance(original_trace_latlon, list) and len(original_trace_latlon) > 0:
            xs_proj_orig = []
            ys_proj_orig = []
            for lat, lon in original_trace_latlon:
                try:
                    x, y = gps_transformer_for_viz.transform(lon, lat)
                    xs_proj_orig.append(x)
                    ys_proj_orig.append(y)
                except: continue
            
            if xs_proj_orig and ys_proj_orig:
                ax.plot(xs_proj_orig, ys_proj_orig, marker='o', linestyle='-', markersize=3, color='blue', alpha=0.6, label='原始GPS轨迹 (投影后)', zorder=2)
                min_x_viz, max_x_viz = min(xs_proj_orig), max(xs_proj_orig)
                min_y_viz, max_y_viz = min(ys_proj_orig), max(ys_proj_orig)
        else:
            plot_title += " (原始轨迹缺失/为空)"

        if matched_path_wkt_str:
            try:
                matched_geometry = wkt.loads(matched_path_wkt_str)
                gpd.GeoSeries([matched_geometry]).plot(ax=ax, linewidth=2.0, color='red', label='LMM匹配路径', zorder=3)
                
                if matched_geometry and not matched_geometry.is_empty:
                    m_bounds = matched_geometry.bounds
                    if len(m_bounds) == 4:
                        m_min_x, m_min_y, m_max_x, m_max_y = m_bounds
                        if min_x_viz is None or m_min_x < min_x_viz: min_x_viz = m_min_x
                        if max_x_viz is None or m_max_x > max_x_viz: max_x_viz = m_max_x
                        if min_y_viz is None or m_min_y < min_y_viz: min_y_viz = m_min_y
                        if max_y_viz is None or m_max_y > max_y_viz: max_y_viz = m_max_y
            except Exception as e:
                # print(f"错误：无法从WKT加载匹配路径为车辆 {vehicle_id}: {e}")
                plot_title += " (匹配路径WKT加载失败)"
        else:
            plot_title += f" (匹配失败或无路径: {result_data.get('error', '未知错误')})"

        if all(v is not None for v in [min_x_viz, max_x_viz, min_y_viz, max_y_viz]):
            padding_x = (max_x_viz - min_x_viz) * 0.10 + 100
            padding_y = (max_y_viz - min_y_viz) * 0.10 + 100
            ax.set_xlim(min_x_viz - padding_x, max_x_viz + padding_x)
            ax.set_ylim(min_y_viz - padding_y, max_y_viz + padding_y)

        ax.set_xlabel(f"X坐标 (米, {VISUALIZATION_TARGET_CRS})")
        ax.set_ylabel(f"Y坐标 (米, {VISUALIZATION_TARGET_CRS})")
        ax.set_title(plot_title, fontsize=10)
        ax.legend(loc='best', fontsize='small')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    print(f"可视化耗时 {time.time() - visualization_start_time:.2f} 秒")
    print(f"--- 总脚本执行耗时: {time.time() - overall_start_time:.2f} 秒 ---")

