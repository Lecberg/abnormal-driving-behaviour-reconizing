# hmm_matcher.py
import math
import numpy as np
from shapely.geometry import Point
from rtree import index # 导入rtree

# --- HMM 模型参数 (可以作为函数参数传递，或在此处定义为默认值) ---
# 这些可以移到主脚本的配置部分，然后传递给 viterbi_map_matching
# SIGMA_OBS_DEFAULT = 30.0
# BETA_DEFAULT = 30.0
# SEARCH_RADIUS_DEFAULT = 100.0

# --- HMM 辅助函数 ---

def point_to_segment_projection(point_coords, segment_geom):
    """
    Calculates the projection of a point onto a line segment and the distance.
    """
    point = Point(point_coords)
    # projected_point_on_line = segment_geom.interpolate(segment_geom.project(point))
    # distance = point.distance(projected_point_on_line)
    
    # 更稳健的方式，直接计算点到线的距离
    distance = point.distance(segment_geom)
    # project() 返回沿线的距离，interpolate() 将此距离转换为点
    projected_point_on_line = segment_geom.interpolate(segment_geom.project(point, normalized=False), normalized=False)
    return projected_point_on_line, distance

def great_circle_distance(p1_coords, p2_coords):
    """
    Calculates Euclidean distance between two projected points (approximates great circle for local areas).
    Assumes coordinates are already in a projected CRS (e.g., UTM).
    """
    return math.sqrt((p1_coords[0] - p2_coords[0])**2 + (p1_coords[1] - p2_coords[1])**2)

def get_route_distance(proj1_on_s1, s1_id, proj2_on_s2, s2_id, road_network):
    """
    Calculates the route distance between two projected points on segments.
    Simplified: considers same segment or directly connected segments.
    """
    s1_geom = road_network[s1_id]["geom"]
    s2_geom = road_network[s2_id]["geom"]

    dist_proj1_on_s1 = s1_geom.project(Point(proj1_on_s1.coords[0]))
    dist_proj2_on_s2 = s2_geom.project(Point(proj2_on_s2.coords[0]))

    if s1_id == s2_id:
        return abs(dist_proj1_on_s1 - dist_proj2_on_s2)
    
    # Check for direct connection (end of s1 is start of s2)
    if road_network[s1_id]["end_node"] == road_network[s2_id]["start_node"]:
        # Distance from projection on s1 to end of s1
        dist_on_s1_to_end = s1_geom.length - dist_proj1_on_s1
        # Distance from start of s2 to projection on s2
        dist_on_s2_from_start = dist_proj2_on_s2
        return dist_on_s1_to_end + dist_on_s2_from_start
    
    # Add check for reverse connection if your network has s1.start_node == s2.end_node
    # and you want to consider that as a valid (but perhaps penalized) transition.
    # For now, only forward connection is considered for simplicity.

    return float('inf') # No simple route found

def get_candidate_segments(gps_point_coords, road_network, road_idx, search_radius):
    """
    Finds candidate road segments near a GPS point using a spatial index.
    """
    point = Point(gps_point_coords)
    query_bounds = (
        point.x - search_radius,
        point.y - search_radius,
        point.x + search_radius,
        point.y + search_radius
    )
    
    # Query spatial index for segments whose bounding boxes intersect the query bounds
    # The `obj` stored during index insertion was the segment_id
    candidate_seg_ids_from_index = [n.object for n in road_idx.intersection(query_bounds, objects=True)]

    candidates = []
    if not candidate_seg_ids_from_index: # Initial broad check found nothing
        # Fallback: check a few closest segments if index yields nothing (could be due to large radius vs. sparse area)
        # This part is optional and can be computationally expensive if road_network is huge.
        # A better fallback might be to increase search_radius slightly for the index query.
        # For now, we'll stick to the original fallback of taking the first few if candidates is empty.
        pass

    for seg_id in candidate_seg_ids_from_index:
        # Ensure seg_id is valid and exists in the road_network
        if seg_id not in road_network:
            # This might happen if the index is out of sync or contains stale IDs
            # print(f"Warning: seg_id {seg_id} from spatial index not in road_network.")
            continue
        segment_geom = road_network[seg_id]["geom"]
        if point.distance(segment_geom) <= search_radius:
            candidates.append(seg_id)
            
    if not candidates:
        # Fallback if no candidates found within radius after precise check,
        # or if index query returned nothing.
        # This ensures the HMM always has some states to start with.
        # Consider if this fallback is always desirable.
        # A more robust HMM might handle "no candidates" by assigning very low probability or a special state.
        # print(f"Warning: No candidates found for point {gps_point_coords} within radius {search_radius}. Using fallback.")
        
        # A slightly better fallback than just taking the first N:
        # Find the single closest segment if no candidates are within search_radius.
        # This is computationally more expensive if done for every point without candidates.
        # For now, let's keep the simple fallback to avoid adding too much complexity here.
        # If you have a very sparse network or points far from roads, this fallback might be hit often.
        
        # Simple fallback: take the first few segments from the network
        # This is a very crude fallback and might lead to poor matches if the point is far from these.
        # return list(road_network.keys())[:5] # Return a small number of arbitrary segments
        
        # A slightly more refined fallback: if index gave some candidates but distance check failed for all,
        # maybe return the one with the minimum distance from the index candidates, even if > search_radius.
        # However, the current logic is: if index gave candidates, they are checked. If none pass, candidates is empty.
        
        # If candidates list is empty, it means either:
        # 1. Spatial index returned no coarse candidates.
        # 2. Spatial index returned coarse candidates, but none passed the precise distance check.
        # In either case, we need a fallback.
        
        # Fallback: find the single closest segment in the entire network
        # This is slow but ensures at least one candidate.
        # Only do this if absolutely necessary and be aware of performance.
        # For a production system, you might want to log this and investigate why points are so far.
        if not road_network: return [] # Should not happen if road_network is loaded

        min_dist = float('inf')
        closest_seg = None
        for seg_id_fallback, seg_data_fallback in road_network.items():
            dist_fallback = point.distance(seg_data_fallback["geom"])
            if dist_fallback < min_dist:
                min_dist = dist_fallback
                closest_seg = seg_id_fallback
        if closest_seg:
            # print(f"Fallback: Closest segment is {closest_seg} at distance {min_dist:.2f}m")
            return [closest_seg]
        else: # Should not happen if road_network is not empty
            return []


    return candidates


def calculate_emission_probability(gps_point_coords, segment_id, road_network, sigma_obs):
    """
    Calculates the emission probability of a GPS point given a road segment.
    """
    if segment_id not in road_network:
        # print(f"Warning: segment_id {segment_id} not found in road_network for emission calculation.")
        return 1e-9 # Very low probability
    segment_geom = road_network[segment_id]["geom"]
    _, distance = point_to_segment_projection(gps_point_coords, segment_geom)
    
    # Gaussian probability
    prob = (1.0 / (sigma_obs * math.sqrt(2 * math.pi))) * math.exp(- (distance**2) / (2 * sigma_obs**2))
    return prob if prob > 1e-9 else 1e-9 # Floor probability to avoid zero

def calculate_transition_probability(prev_gps_coords, curr_gps_coords,
                                     prev_segment_id, curr_segment_id,
                                     road_network, beta):
    """
    Calculates the transition probability between two segments given two consecutive GPS points.
    """
    if prev_segment_id not in road_network or curr_segment_id not in road_network:
        # print(f"Warning: segment_id not found for transition: {prev_segment_id} or {curr_segment_id}")
        return 1e-9

    prev_segment_geom = road_network[prev_segment_id]["geom"]
    curr_segment_geom = road_network[curr_segment_id]["geom"]

    proj_prev, _ = point_to_segment_projection(prev_gps_coords, prev_segment_geom)
    proj_curr, _ = point_to_segment_projection(curr_gps_coords, curr_segment_geom)

    dist_gps = great_circle_distance(prev_gps_coords, curr_gps_coords)
    if dist_gps < 1e-6: dist_gps = 1e-6 # Avoid division by zero or very small distances

    dist_route = get_route_distance(proj_prev, prev_segment_id, proj_curr, curr_segment_id, road_network)

    if dist_route == float('inf'): # Segments are not "easily" routable in our simplified model
        return 1e-9 # Very low probability for non-routable transitions

    # Difference between GPS distance and route distance
    # The Newson and Krumm paper uses abs(dist_gps - dist_route)
    # The probability is higher if this difference is small
    diff = abs(dist_gps - dist_route)
    
    # Exponential decay based on the difference
    # prob = (1.0 / beta) * math.exp(-diff / beta) # Original formulation from some papers
    # Let's use a slightly different formulation that is less sensitive to beta's exact value
    # and focuses on the ratio of distances.
    # A common approach is to use the ratio, or a penalty for large differences.
    # For now, let's stick to a variation of the exponential decay on the difference.
    
    # Transition probability based on Newson & Krumm (inverse of distance difference)
    # This is simplified. A more robust model would use dist_route / dist_gps or similar.
    # The paper's formula is more like: P(r_i | r_{i-1}) = d_u / d_p
    # where d_u is great circle distance between observations, d_p is path distance.
    # Let's use the exponential decay on the difference, as it's common in HMM map matching.
    prob = math.exp(-diff / beta) # Higher beta means more tolerance for difference

    # Penalize transitions that are not directly connected, even if dist_route was calculated
    # (e.g., if get_route_distance was extended to handle >1 hop paths)
    # The current get_route_distance only returns non-inf for s1==s2 or s1.end==s2.start
    # So this check might be redundant with current get_route_distance but good for future.
    if prev_segment_id != curr_segment_id:
        if road_network[prev_segment_id]["end_node"] != road_network[curr_segment_id]["start_node"]:
            # This case should ideally be handled by get_route_distance returning inf,
            # but as a safeguard or if get_route_distance changes:
            prob *= 0.01 # Heavy penalty for non-connected segments if not handled by route distance

    return prob if prob > 1e-9 else 1e-9


# --- Viterbi Algorithm ---
def viterbi_map_matching(gps_trace, road_network, road_idx, sigma_obs, beta, search_radius):
    """
    Performs map matching using the Viterbi algorithm.
    Args:
        gps_trace (list of tuples): List of (x, y) GPS coordinates.
        road_network (dict): Processed road network data.
        road_idx (rtree.index.Index): Spatial index for the road network.
        sigma_obs (float): Standard deviation for emission probability.
        beta (float): Parameter for transition probability.
        search_radius (float): Radius to search for candidate segments.
    Returns:
        list: List of matched road segment IDs, or error strings.
    """
    num_gps_points = len(gps_trace)
    if num_gps_points == 0:
        return []

    # Viterbi probabilities: V[t][state] = max probability of any path ending at state at time t
    V = [{} for _ in range(num_gps_points)]
    # Path pointers: path[t][state] = previous state that maximizes V[t][state]
    path = [{} for _ in range(num_gps_points)]

    # Initialization step (t=0)
    first_gps_point = gps_trace[0]
    candidate_segments_t0 = get_candidate_segments(first_gps_point, road_network, road_idx, search_radius)

    if not candidate_segments_t0:
        # print(f"No candidate segments found for the first GPS point: {first_gps_point}")
        return ["NO_CANDIDATES_START"] * num_gps_points

    initial_probs_sum = 0
    for seg_id in candidate_segments_t0:
        ep = calculate_emission_probability(first_gps_point, seg_id, road_network, sigma_obs)
        # Initial state probability can be assumed uniform or based on emission
        # Here, we use emission probability (normalized)
        V[0][seg_id] = ep 
        path[0][seg_id] = None # No previous state for t=0
        initial_probs_sum += ep
    
    # Normalize initial probabilities if sum is not zero
    if initial_probs_sum > 1e-9:
        for seg_id in candidate_segments_t0:
            V[0][seg_id] /= initial_probs_sum
    else: # All emission probabilities were tiny, assign uniform
        if candidate_segments_t0: # Check if list is not empty
            uniform_prob = 1.0 / len(candidate_segments_t0)
            for seg_id in candidate_segments_t0:
                V[0][seg_id] = uniform_prob
        # If candidate_segments_t0 was empty, this branch won't be hit due to earlier check.


    # Recursion step (t=1 to num_gps_points-1)
    for t in range(1, num_gps_points):
        curr_gps_point = gps_trace[t]
        prev_gps_point = gps_trace[t-1]
        
        candidate_segments_curr_t = get_candidate_segments(curr_gps_point, road_network, road_idx, search_radius)
        
        if not candidate_segments_curr_t:
            # print(f"No candidates for GPS point {t} at {curr_gps_point}. Attempting to carry forward previous best.")
            # Handle missing candidates: try to carry forward from previous step or mark as error
            # A simple strategy: if previous step had a best, assume it persists with a penalty.
            # This is a heuristic. More advanced methods might try to "jump" or re-initialize.
            if V[t-1]: # If previous step had valid states
                # Find the best segment from the previous step
                # Default to None if V[t-1] is empty (should not happen if t0 had candidates)
                best_prev_seg_overall = max(V[t-1], key=V[t-1].get, default=None) 
                if best_prev_seg_overall:
                    # Carry forward this segment with a penalty (e.g., low transition prob)
                    # Emission prob for curr_gps_point to best_prev_seg_overall
                    ep_carry = calculate_emission_probability(curr_gps_point, best_prev_seg_overall, road_network, sigma_obs)
                    V[t][best_prev_seg_overall] = V[t-1][best_prev_seg_overall] * 0.01 * ep_carry # Small transition, times emission
                    path[t][best_prev_seg_overall] = best_prev_seg_overall # Points to itself
                else: # V[t-1] was empty, major issue
                    # print(f"Critical: V[t-1] empty at t={t}, cannot proceed.")
                    return ["ERROR_VITERBI_EMPTY_PREV_STATE"] * num_gps_points
            else: # V[t-1] was empty
                # print(f"Critical: V[t-1] empty at t={t}, cannot proceed (initial error).")
                return ["ERROR_VITERBI_EMPTY_PREV_STATE_INIT"] * num_gps_points
            continue # Move to next GPS point

        possible_prev_segments = list(V[t-1].keys()) # Segments that had a path at t-1

        for curr_seg_id in candidate_segments_curr_t:
            max_prob_for_curr_seg = -1.0
            best_prev_seg_for_curr = None
            
            ep_curr = calculate_emission_probability(curr_gps_point, curr_seg_id, road_network, sigma_obs)
            if ep_curr < 1e-9: # If emission is virtually zero, skip complex transition calcs for this candidate
                # V[t][curr_seg_id] = 1e-12 # Assign a tiny probability
                # path[t][curr_seg_id] = None # Or point to a default if needed
                continue


            for prev_seg_id in possible_prev_segments:
                if V[t-1].get(prev_seg_id, 0) < 1e-9: # Skip if previous path prob was negligible
                    continue

                tp = calculate_transition_probability(prev_gps_point, curr_gps_point,
                                                      prev_seg_id, curr_seg_id,
                                                      road_network, beta)
                
                current_path_prob = V[t-1][prev_seg_id] * tp * ep_curr
                
                if current_path_prob > max_prob_for_curr_seg:
                    max_prob_for_curr_seg = current_path_prob
                    best_prev_seg_for_curr = prev_seg_id
            
            if best_prev_seg_for_curr is not None:
                V[t][curr_seg_id] = max_prob_for_curr_seg
                path[t][curr_seg_id] = best_prev_seg_for_curr
            # else:
                # If no prev_seg_id led to a positive probability (e.g., all TPs were zero)
                # This curr_seg_id might not be reachable.
                # We could assign a very small probability based on emission only,
                # or let it remain unassigned if max_prob_for_curr_seg is still -1.0.
                # If ep_curr was high, but all transitions were bad:
                # V[t][curr_seg_id] = ep_curr * 1e-6 # Penalized emission-only path
                # path[t][curr_seg_id] = None # No valid predecessor based on transition
                # Current logic: if max_prob_for_curr_seg remains -1, this state won't be in V[t]
                # unless ep_curr was high and we explicitly add it.
                # The `if ep_curr < 1e-9: continue` handles cases where emission is too low.
                # If ep_curr is good, but all transitions are bad, best_prev_seg_for_curr will be None.
                # In this case, the segment curr_seg_id will not be added to V[t] for this t.
                # This is generally fine, as it means it's not a likely continuation.

    # Termination and Path Backtracking
    last_valid_t = -1
    for i in range(num_gps_points - 1, -1, -1):
        if V[i]: # Find the last time step with any valid states
            last_valid_t = i
            break
    
    if last_valid_t == -1: # No valid path found through the entire trace
        # print("No valid path found by Viterbi algorithm.")
        return ["NO_VALID_VITERBI_PATH"] * num_gps_points

    # Find the best segment at the last valid time step
    # Default to None if V[last_valid_t] is empty (should not happen if last_valid_t != -1)
    best_last_segment_id = max(V[last_valid_t], key=V[last_valid_t].get, default=None)

    if best_last_segment_id is None:
        # This case implies V[last_valid_t] was empty, which contradicts last_valid_t check.
        # Or, all probabilities in V[last_valid_t] were non-numeric or problematic.
        # print("Error: Could not determine best last segment despite last_valid_t.")
        return ["ERROR_BEST_LAST_SEG_UNDETERMINED"] * num_gps_points

    # Initialize matched_path with a placeholder
    matched_path_ids = ["UNMATCHED"] * num_gps_points 
    
    # Backtrack from last_valid_t
    current_seg_for_backtrack = best_last_segment_id
    for t_idx in range(last_valid_t, -1, -1):
        matched_path_ids[t_idx] = current_seg_for_backtrack
        if t_idx > 0:
            prev_seg_in_path = path[t_idx].get(current_seg_for_backtrack)
            if prev_seg_in_path is None:
                # Path broken during backtracking. This can happen if a state was reached
                # via a "carry-forward" or an emission-only path without a proper predecessor.
                # print(f"Path broken during backtracking at t={t_idx} for segment {current_seg_for_backtrack}.")
                # Attempt to fill remaining earlier points with the current segment or a special marker.
                fill_value = matched_path_ids[t_idx] # Use the last known good segment
                for k_fill_back in range(t_idx - 1, -1, -1):
                    if matched_path_ids[k_fill_back] == "UNMATCHED": # Only fill if not already filled
                         matched_path_ids[k_fill_back] = fill_value
                    # else: # If already filled by a previous break, respect that.
                    #     break 
                break # Stop backtracking along this broken path
            current_seg_for_backtrack = prev_seg_in_path
        else: # t_idx == 0
            break 

    # Handle points after last_valid_t (if any, due to no candidates at later stages)
    # Fill them with the best_last_segment_id
    if last_valid_t < num_gps_points - 1 and best_last_segment_id:
        for t_fill_forward in range(last_valid_t + 1, num_gps_points):
            if matched_path_ids[t_fill_forward] == "UNMATCHED":
                matched_path_ids[t_fill_forward] = best_last_segment_id
    
    # Final check for any remaining "UNMATCHED" (e.g., if first point had no candidates)
    # This should be rare if initial checks are robust.
    # A simple forward fill from the first valid match, or backward fill from last.
    # For now, we assume the backtracking and forward fill cover most cases.
    # If matched_path_ids[0] is "UNMATCHED" and others are matched, fill it.
    if matched_path_ids[0] == "UNMATCHED":
        first_known_good = next((seg for seg in matched_path_ids if seg != "UNMATCHED"), None)
        if first_known_good:
            for i in range(len(matched_path_ids)):
                if matched_path_ids[i] == "UNMATCHED":
                    matched_path_ids[i] = first_known_good
                else:
                    break # Stop once we hit the start of the known good sequence

    return matched_path_ids
