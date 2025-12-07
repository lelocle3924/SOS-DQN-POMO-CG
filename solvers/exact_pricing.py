import numpy as np
import heapq
from configs import main_config as cfg

# Performance Tuned
MAX_EXACT_ITERATIONS = cfg.EXACT_MAX_ITE
NEIGHBOR_LIMIT = 20

class Label:
    """Lightweight State Object"""
    __slots__ = ('curr_node', 'rc', 'cost', 'time', 'load', 'visited_mask', 'parent')
    
    def __init__(self, curr_node, rc, cost, time, load, visited_mask, parent):
        self.curr_node = int(curr_node)
        self.rc = float(rc)
        self.cost = float(cost)
        self.time = float(time)
        self.load = float(load)
        self.visited_mask = int(visited_mask)
        self.parent = parent

    def __lt__(self, other):
        return self.rc < other.rc

    def dominates(self, other):
        # Reduced Cost (Objective) thấp hơn
        if self.rc > other.rc: return False
        # Resource consumption thấp hơn
        if self.time > other.time: return False
        if self.load > other.load: return False
        
        # Elementary check: Self visited tập con của Other
        # bitwise: (Self | Other) == Other -> Self subset Other
        if (self.visited_mask | other.visited_mask) != other.visited_mask:
            return False
        return True

class ExactPricingSolver:
    def __init__(self, locations_df, dist_matrix, time_matrix):
        self.locations_df = locations_df
        # Mapping: CustomerID (Real) -> Matrix Index (0..N)
        # Giả định: dist_matrix xếp theo thứ tự rows của locations_df
        # Index 0 luôn là Depot
        self.dist_matrix = dist_matrix 
        self.time_matrix = time_matrix 
        
        self.num_nodes = len(locations_df)
        self.capacity_limit = cfg.VEHICLE_CAPACITY
        
        # Cache Columns
        self.node_ids = locations_df['Number'].values
        self.demands = locations_df['Demand'].values
        self.tw_start = locations_df['Start_Time'].values
        self.tw_end = locations_df['End_Time'].values
        self.service_times = locations_df['ServiceTime'].values
        
        # Mapping ngược ID -> Index để lấy Duals cho nhanh
        self.id_to_idx = {nid: idx for idx, nid in enumerate(self.node_ids)}
        
        self.neighbors = self._precompute_neighbors()

    def _precompute_neighbors(self):
        """Build Reduced Graph (Adjacency List)"""
        nbs = {}
        for i in range(self.num_nodes):
            # Sort node by time distance
            dists = self.time_matrix[i]
            sorted_idx = np.argsort(dists)
            
            node_nbs = []
            count = 0
            for v in sorted_idx:
                if v == i: continue
                # Simple pruning: 1-step unreachable TW
                # depart i = earliest start i + service
                arrival = self.tw_start[i] + self.service_times[i] + self.time_matrix[i, v]
                if arrival > self.tw_end[v]:
                    continue
                
                node_nbs.append(v)
                if v != 0: count += 1 
                
                if count >= NEIGHBOR_LIMIT: 
                    break
            
            # Depot back-link is mandatory
            if i != 0 and 0 not in node_nbs:
                node_nbs.append(0)
            
            nbs[i] = node_nbs
        return nbs

    def solve(self, duals_dict, gap_tolerance=-1e-4):
        """
        Label Setting Algorithm
        Returns: list[list[int]] (routes)
        """
        # 1. Map Duals dict -> Array
        duals_arr = np.zeros(self.num_nodes)
        for nid, d_val in duals_dict.items():
            if nid in self.id_to_idx:
                idx = self.id_to_idx[nid]
                duals_arr[idx] = d_val

        # 2. Init Label at Depot
        start_label = Label(curr_node=0, rc=0.0, cost=0.0, 
                            time=self.tw_start[0], load=0.0, 
                            visited_mask=1, parent=None)
        
        # Priority Queue (Min Heap by Reduced Cost)
        queue = [start_label]
        
        # Labels Registry for Dominance [Node_Index] -> List[Label]
        labels_registry = {i: [] for i in range(self.num_nodes)}
        
        best_end_label = None
        iters = 0
        
        while queue:
            curr = heapq.heappop(queue)
            u = curr.curr_node
            
            # Expand
            for v in self.neighbors[u]:
                # A. Resource Check
                
                # 1. Elementary
                if v != 0 and ((curr.visited_mask >> v) & 1):
                    continue
                
                # 2. Capacity
                new_load = curr.load + self.demands[v]
                if new_load > self.capacity_limit: continue
                
                # 3. Time Windows
                arrival = curr.time + self.service_times[u] + self.time_matrix[u, v]
                if arrival > self.tw_end[v]: continue
                start_service_v = max(arrival, self.tw_start[v])
                
                # B. Cost Check
                # RC_new = RC_prev + cost_uv - dual_v
                # Depot (0) dual = 0
                step_cost = self.time_matrix[u, v]
                dual_v = duals_arr[v] if v != 0 else 0.0
                new_rc = curr.rc + step_cost - dual_v
                new_cost = curr.cost + step_cost
                
                # New mask
                new_mask = curr.visited_mask | (1 << v) if v != 0 else curr.visited_mask
                
                new_lbl = Label(v, new_rc, new_cost, start_service_v, new_load, new_mask, curr)
                
                # C. Check Dominance at node v
                is_dominated = False
                existing_list = labels_registry[v]
                # Filter out dominated labels
                survivor_labels = []
                
                for ex in existing_list:
                    if ex.dominates(new_lbl):
                        is_dominated = True
                        break
                    if not new_lbl.dominates(ex):
                        survivor_labels.append(ex)
                
                if is_dominated: 
                    continue
                
                # D. Store & Push
                labels_registry[v] = survivor_labels + [new_lbl]
                
                if v == 0:
                    # Closing Route
                    if new_lbl.rc < gap_tolerance:
                        if best_end_label is None or new_lbl.rc < best_end_label.rc:
                            best_end_label = new_lbl
                else:
                    heapq.heappush(queue, new_lbl)
            
            iters += 1
            if iters > MAX_EXACT_ITERATIONS:
                print(f"[ExactPricing] Max iterations ({MAX_EXACT_ITERATIONS}) reached.")
                break
                
        # Reconstruct Best Route
        if best_end_label:
            # Map Indices back to Real IDs (if necessary).
            # Here we reconstruct Index Route [0, idx1, idx2, 0]
            # Must map Index -> Real ID using self.node_ids
            path = []
            ptr = best_end_label
            while ptr:
                idx = ptr.curr_node
                real_id = int(self.node_ids[idx])
                path.append(real_id)
                ptr = ptr.parent
            
            final_route = path[::-1] # Reverse
            print(f"[ExactPricing] Found optimal column with RC={best_end_label.rc:.2f}")
            return [final_route]
            
        return [] # <--- DÒNG BẠN THIẾU