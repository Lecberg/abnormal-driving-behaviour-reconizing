# gps_map_matching3.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from pyproj import Transformer
import matplotlib.pyplot as plt
import random
import time # For timing
import os

# --- Import HMM functions from hmm_matcher.py ---
from hmm_matcher import (
    viterbi_map_matching # We only need to import the main Viterbi function here
                         # Helper functions are used internally by viterbi_map_matching
)
# For spatial index, rtree is imported within hmm_matcher.py and load_and_preprocess_roads
from rtree import index as rtree_index # Need this for type hinting or direct use if any

# --- Multiprocessing ---
import multiprocessing as mp
from functools import partial

# --- 1. 配置 ---
ROAD_NETWORK_FILE = 'map_matching/610100路网.geojson'
GPS_TRAJECTORIES_FILE = 'map_matching/gps_features.csv'
TARGET_CRS_EPSG = 32649
SOURCE_CRS_GPS = "EPSG:4326"

DRIVABLE_FCLASSES = [
    'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
    'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified',
    'residential', 'living_street', 'service'
]

# HMM 模型参数 (这些将传递给匹配函数)
SIGMA_OBS = 30.0
BETA = 30.0
SEARCH_RADIUS = 100.0 # Meters

# --- 2. 加载和预处理路网数据 (with Spatial Index) ---
def load_and_preprocess_roads(filepath, target_crs_epsg, drivable_fclasses):
    print(f"Loading road network from {filepath}...")
    try:
        gdf_roads = gpd.read_file(filepath)
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None, None # Return None for network and index
    
    print(f"Original road network CRS: {gdf_roads.crs}")
    if gdf_roads.crs is None:
        print("Warning: Road network CRS is None. Assuming EPSG:4326 (WGS84).")
        gdf_roads.set_crs(f"EPSG:4326", inplace=True, allow_override=True)

    print(f"Projecting road network to EPSG:{target_crs_epsg}...")
    gdf_roads_proj = gdf_roads.to_crs(f"EPSG:{target_crs_epsg}")
    print(f"Projected road network CRS: {gdf_roads_proj.crs}")

    if 'fclass' in gdf_roads_proj.columns:
        print(f"Filtering for drivable roads (fclass in {drivable_fclasses})...")
        original_count = len(gdf_roads_proj)
        gdf_roads_proj = gdf_roads_proj[gdf_roads_proj['fclass'].isin(drivable_fclasses)]
        print(f"Filtered from {original_count} to {len(gdf_roads_proj)} drivable segments.")
    else:
        print("Warning: 'fclass' column not found. Cannot filter drivable roads.")

    processed_roads = {}
    segment_id_counter = 0
    node_coords_to_id = {}
    next_node_id_val = 0

    def get_node_id_local(coords_tuple, precision=2):
        nonlocal next_node_id_val
        rounded_coords = tuple(round(c, precision) for c in coords_tuple)
        if rounded_coords not in node_coords_to_id:
            node_coords_to_id[rounded_coords] = f"node_{next_node_id_val}"
            next_node_id_val += 1
        return node_coords_to_id[rounded_coords]

    for index_row, row in gdf_roads_proj.iterrows(): # Renamed index to index_row
        geom = row['geometry']
        properties = row.drop('geometry').to_dict() # Get properties as dict

        if isinstance(geom, MultiLineString):
            linestrings = list(geom.geoms)
        elif isinstance(geom, LineString):
            linestrings = [geom]
        else:
            continue

        for ls_geom in linestrings:
            if not isinstance(ls_geom, LineString) or ls_geom.is_empty or len(list(ls_geom.coords)) < 2:
                continue
            
            segment_id_base = f"road_s{segment_id_counter}"
            segment_id_counter += 1

            start_node_coords = ls_geom.coords[0]
            end_node_coords = ls_geom.coords[-1]
            start_node_id = get_node_id_local(start_node_coords)
            end_node_id = get_node_id_local(end_node_coords)

            oneway_status = str(properties.get('oneway', 'F')).upper()
            
            current_osm_id = properties.get('osm_id', index_row) # Use GeoDataFrame index if osm_id missing

            processed_roads[segment_id_base] = {
                "geom": ls_geom, "start_node": start_node_id, "end_node": end_node_id,
                "osm_id": current_osm_id, "name": properties.get('name', 'N/A'),
                "fclass": properties.get('fclass', 'N/A'), "oneway_orig": oneway_status,
                "length": ls_geom.length
            }

            if oneway_status in ['F', 'NO', '0', 'FALSE', ''] or oneway_status is None:
                segment_id_rev = f"{segment_id_base}_rev"
                reversed_coords = list(ls_geom.coords)[::-1]
                ls_geom_reverse = LineString(reversed_coords)
                processed_roads[segment_id_rev] = {
                    "geom": ls_geom_reverse, "start_node": end_node_id, "end_node": start_node_id,
                    "osm_id": current_osm_id, "name": properties.get('name', 'N/A'),
                    "fclass": properties.get('fclass', 'N/A'), "oneway_orig": oneway_status,
                    "length": ls_geom_reverse.length
                }
            elif oneway_status in ['B', '-1']:
                current_seg_data = processed_roads[segment_id_base]
                current_seg_data["geom"] = LineString(list(ls_geom.coords)[::-1])
                current_seg_data["start_node"] = end_node_id
                current_seg_data["end_node"] = start_node_id
    
    print(f"Processed into {len(processed_roads)} HMM-ready road segments.")
    print(f"Created {len(node_coords_to_id)} unique nodes using {next_node_id_val} IDs.")

    # --- Build Spatial Index ---
    print("Building spatial index for road segments...")
    spatial_idx = rtree_index.Index()
    # Store an integer id for rtree, and the actual segment_id as the object
    for i, seg_id_key in enumerate(processed_roads.keys()):
        # Ensure geometry is valid and has bounds
        if processed_roads[seg_id_key]['geom'] and not processed_roads[seg_id_key]['geom'].is_empty:
            spatial_idx.insert(i, processed_roads[seg_id_key]['geom'].bounds, obj=seg_id_key)
        else:
            print(f"Warning: Skipping segment {seg_id_key} in spatial index due to invalid/empty geometry.")

    print("Spatial index built.")
    return processed_roads, spatial_idx


# --- 3. 加载和预处理GPS轨迹数据 ---
def load_and_preprocess_gps(filepath, source_crs, target_crs_epsg):
    print(f"Loading GPS trajectories from {filepath}...")
    try:
        df_gps = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}

    required_cols = ['vid_md5', 'Lng', 'Lat']
    if not all(col in df_gps.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return {}

    time_col_found = False
    for col_name in ['标准时间', 'timestamp', 'time']: # Check for common time column names
        if col_name in df_gps.columns:
            try:
                print(f"Sorting by time column: {col_name}")
                df_gps[col_name] = pd.to_datetime(df_gps[col_name])
                df_gps.sort_values(by=['vid_md5', col_name], inplace=True)
                time_col_found = True
                break 
            except Exception as e:
                print(f"Warning: Could not parse or sort by time column {col_name}: {e}")
    if not time_col_found:
        print("Warning: No recognized time column found for sorting trajectories.")


    print(f"Projecting GPS data from {source_crs} to EPSG:{target_crs_epsg}...")
    transformer = Transformer.from_crs(source_crs, f"EPSG:{target_crs_epsg}", always_xy=True)
    
    projected_trajectories = {}
    for vehicle_id, group in df_gps.groupby('vid_md5'):
        trajectory_points = []
        for _, row in group.iterrows():
            lon, lat = row['Lng'], row['Lat']
            try:
                x, y = transformer.transform(lon, lat)
                trajectory_points.append((x, y))
            except Exception as e:
                # print(f"Warning: Could not project point ({lon}, {lat}) for vehicle {vehicle_id}: {e}")
                continue
        if trajectory_points:
            projected_trajectories[vehicle_id] = trajectory_points
            
    print(f"Loaded and projected trajectories for {len(projected_trajectories)} vehicles.")
    return projected_trajectories

# --- Worker function for multiprocessing ---
# This function will be called by each process.
# It needs all necessary data for a single vehicle's matching.
def match_vehicle_trajectory_worker(vehicle_data_tuple, road_network_ref, road_idx_ref, 
                                    sigma_obs_ref, beta_ref, search_radius_ref):
    vehicle_id, gps_trace_proj = vehicle_data_tuple
    # print(f"  Starting matching for vehicle: {vehicle_id} in process {os.getpid()}")
    
    if not gps_trace_proj or len(gps_trace_proj) < 2 : # Need at least 2 points for HMM transitions
        # print(f"    Skipping vehicle {vehicle_id} due to empty or too short projected trace.")
        return vehicle_id, {"original_proj": gps_trace_proj, "matched_segment_ids": ["TRACE_TOO_SHORT"]}

    matched_segment_ids = viterbi_map_matching(
        gps_trace_proj,
        road_network_ref,
        road_idx_ref,
        sigma_obs_ref,
        beta_ref,
        search_radius_ref
    )
    # print(f"    Finished matching for vehicle: {vehicle_id} in process {os.getpid()}. Matched {len(matched_segment_ids)} points.")
    return vehicle_id, {"original_proj": gps_trace_proj, "matched_segment_ids": matched_segment_ids}

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    overall_start_time = time.time()

    # --- Load Data ---
    print("--- Loading and Preprocessing Road Network ---")
    load_roads_start_time = time.time()
    road_network_hmm, road_spatial_index = load_and_preprocess_roads(
        ROAD_NETWORK_FILE, TARGET_CRS_EPSG, DRIVABLE_FCLASSES
    )
    print(f"Road network loading and preprocessing took {time.time() - load_roads_start_time:.2f}s")

    print("\n--- Loading and Preprocessing GPS Trajectories ---")
    load_gps_start_time = time.time()
    all_gps_trajectories_proj = load_and_preprocess_gps(
        GPS_TRAJECTORIES_FILE, SOURCE_CRS_GPS, TARGET_CRS_EPSG
    )
    print(f"GPS data loading and preprocessing took {time.time() - load_gps_start_time:.2f}s")

    if not road_network_hmm or not road_spatial_index or not all_gps_trajectories_proj:
        print("\nExiting due to errors in data loading or preprocessing.")
        exit()

    all_vehicle_ids = list(all_gps_trajectories_proj.keys())
    if not all_vehicle_ids:
        print("\nNo GPS trajectories to process.")
        exit()
        
    print(f"\n--- Performing Map Matching for {len(all_vehicle_ids)} vehicles ---")
    matching_start_time = time.time()
    
    results = {}
    
    # Prepare data for multiprocessing
    # Tuples of (vehicle_id, trajectory_data)
    tasks = [(vid, all_gps_trajectories_proj[vid]) for vid in all_vehicle_ids if all_gps_trajectories_proj[vid]]

    # Use functools.partial to create a new function with some arguments pre-filled.
    # This is cleaner for Pool.map if the worker function takes many static arguments.
    # The worker function `match_vehicle_trajectory_worker` is defined to accept these.
    # The first argument to worker (vehicle_data_tuple) will come from `tasks`.
    partial_worker = partial(match_vehicle_trajectory_worker,
                             road_network_ref=road_network_hmm,
                             road_idx_ref=road_spatial_index,
                             sigma_obs_ref=SIGMA_OBS,
                             beta_ref=BETA,
                             search_radius_ref=SEARCH_RADIUS)

    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} processes for matching.")

    with mp.Pool(processes=num_processes) as pool:
        # pool.map expects a function that takes a single argument from the iterable (tasks)
        # Our partial_worker now fits this signature.
        # Each item in `processed_results_list` will be a tuple: (vehicle_id, match_output_dict)
        processed_results_list = pool.map(partial_worker, tasks)

    # Collect results from the list of tuples
    for vehicle_id, match_output in processed_results_list:
        results[vehicle_id] = match_output
            
    print(f"Map matching completed for {len(results)} vehicles.")
    print(f"Total matching time: {time.time() - matching_start_time:.2f}s")

    # --- 5. 可视化 ---
    print("\n--- Visualizing Results for Selected Vehicles ---")
    visualization_start_time = time.time()

    if not results:
        print("No matching results to visualize.")
        exit()

    num_vehicles_to_plot = min(5, len(results))
    if num_vehicles_to_plot == 0:
        print("No vehicles with results to plot.")
        exit()
    
    available_vehicle_ids_with_results = [
        vid for vid, res_data in results.items() 
        if res_data and "matched_segment_ids" in res_data and 
           (not isinstance(res_data["matched_segment_ids"], list) or 
            (res_data["matched_segment_ids"] and res_data["matched_segment_ids"][0] not in ["TRACE_TOO_SHORT", "NO_CANDIDATES_START"]))
    ]


    if not available_vehicle_ids_with_results:
        print("No vehicles with valid matching results to plot.")
        exit()
        
    vehicle_ids_to_plot = random.sample(available_vehicle_ids_with_results, 
                                        min(num_vehicles_to_plot, len(available_vehicle_ids_with_results)))
    
    print(f"Will generate individual plots for: {vehicle_ids_to_plot}")

    base_road_geoms = [data['geom'] for seg_id, data in road_network_hmm.items() 
                       if not seg_id.endswith("_rev") and data['geom'] and not data['geom'].is_empty]
    gs_roads = gpd.GeoSeries(base_road_geoms) if base_road_geoms else None

    for i, vehicle_id in enumerate(vehicle_ids_to_plot):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        data = results[vehicle_id]
        original_trace_proj = data.get("original_proj")
        matched_ids = data.get("matched_segment_ids")

        if gs_roads is not None:
            gs_roads.plot(ax=ax, linewidth=0.8, color='lightgray', alpha=0.9, zorder=1)

        plot_title = f"Map Matching: Vehicle {vehicle_id}"
        min_x, max_x, min_y, max_y = None, None, None, None

        if original_trace_proj and isinstance(original_trace_proj, list) and len(original_trace_proj) > 0:
            xs_orig = [p[0] for p in original_trace_proj]
            ys_orig = [p[1] for p in original_trace_proj]
            ax.plot(xs_orig, ys_orig, marker='o', linestyle='-', markersize=3, color='blue', alpha=0.6, label='Original GPS', zorder=2)
            
            if xs_orig and ys_orig:
                min_x, max_x = min(xs_orig), max(xs_orig)
                min_y, max_y = min(ys_orig), max(ys_orig)
        else:
            plot_title += " (Original Trace Missing/Empty)"


        matched_route_geoms = []
        if matched_ids and isinstance(matched_ids, list) and \
           all(isinstance(mid, str) for mid in matched_ids if mid is not None and "ERROR" not in mid and "UNMATCHED" not in mid and "NO_CANDIDATES" not in mid and "TRACE_TOO_SHORT" not in mid):
            
            unique_ordered_matched_ids = []
            last_id = None
            for mid in matched_ids:
                if mid in road_network_hmm and mid != last_id: # Ensure mid is a valid key
                    unique_ordered_matched_ids.append(mid)
                    last_id = mid
            
            for seg_id in unique_ordered_matched_ids:
                if road_network_hmm[seg_id]['geom'] and not road_network_hmm[seg_id]['geom'].is_empty:
                     matched_route_geoms.append(road_network_hmm[seg_id]['geom'])
            
            if matched_route_geoms:
                gs_matched_route = gpd.GeoSeries(matched_route_geoms)
                gs_matched_route.plot(ax=ax, linewidth=2.0, color='red', label='Matched Route', zorder=3)
                
                # Update bounds if matched route extends beyond original
                if gs_matched_route.total_bounds is not None and len(gs_matched_route.total_bounds) == 4 :
                    m_min_x, m_min_y, m_max_x, m_max_y = gs_matched_route.total_bounds
                    if min_x is None or m_min_x < min_x: min_x = m_min_x
                    if max_x is None or m_max_x > max_x: max_x = m_max_x
                    if min_y is None or m_min_y < min_y: min_y = m_min_y
                    if max_y is None or m_max_y > max_y: max_y = m_max_y
            else:
                 plot_title += " (No Valid Matched Geoms)"
        else:
            plot_title += f" (Matching Failed/Invalid: {str(matched_ids)[:30]}...)"


        if all(v is not None for v in [min_x, max_x, min_y, max_y]):
            padding_x = (max_x - min_x) * 0.10 + 50 # 10% + 50m fixed
            padding_y = (max_y - min_y) * 0.10 + 50
            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)
        else: # Fallback if bounds could not be determined
            if gs_roads is not None and not gs_roads.empty:
                ax.set_xlim(gs_roads.total_bounds[0], gs_roads.total_bounds[2])
                ax.set_ylim(gs_roads.total_bounds[1], gs_roads.total_bounds[3])


        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(plot_title + f"\n(EPSG:{TARGET_CRS_EPSG})", fontsize=10)
        ax.legend(loc='best', fontsize='small')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # plt.savefig(f"map_matching_vehicle_{vehicle_id}.png", dpi=200)
        # plt.close(fig)
        plt.show()

    print(f"Visualization took {time.time() - visualization_start_time:.2f}s")
    print(f"--- Total script execution time: {time.time() - overall_start_time:.2f}s ---")

