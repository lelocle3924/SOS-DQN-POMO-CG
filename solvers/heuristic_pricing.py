import torch
import numpy as np
from configs import main_config as cfg
from agents.pomo_generator import POMOGenerator
from solvers.exact_pricing import ExactPricingSolver

class PomoPricingAgent:
    def __init__(self, locations_df, dist_matrix_tensor):
        self.device = cfg.DEVICE
        self.locations_df = locations_df
        
        # 1. Modules
        self.pomo_gen = POMOGenerator(self.device)
        time_matrix_np = dist_matrix_tensor.cpu().numpy()
        self.exact_solver = ExactPricingSolver(locations_df, time_matrix_np, time_matrix_np)
        
        # 2. Compute Static Scale Factors
        coords = locations_df[['Longitude', 'Latitude']].values.astype(np.float32)
        
        self.min_coord = coords.min(axis=0)
        self.max_coord = coords.max(axis=0)
        
        self.scale_coord = np.max(self.max_coord - self.min_coord)
        if self.scale_coord == 0: self.scale_coord = 1.0
        
        # Time Normalization
        depot = locations_df[locations_df['Number'] == 0].iloc[0]
        self.max_time = float(depot['End_Time'])
        if self.max_time <= 1.0: self.max_time = 1440.0
        
        self.problem_size = len(locations_df) - 1
        
        # Prepare Static Tensor
        self._prepare_pomo_tensors(locations_df)
        self.travel_times_tensor = dist_matrix_tensor.unsqueeze(0)

    def _prepare_pomo_tensors(self, df):
        """Normalize dữ liệu Static sang khoảng [0, 1]"""
        depot = df[df['Number'] == 0].iloc[0]
        customers = df[df['Number'] != 0]
        
        # 1. Normalize Coordinates (Lat/Lon -> 0..1)
        # Sửa ở đây để đảm bảo shape đầu ra đúng chuẩn
        
        # Lấy value dưới dạng array (2,)
        depot_raw = depot[['Longitude', 'Latitude']].values.astype(np.float32) 
        node_raw = customers[['Longitude', 'Latitude']].values.astype(np.float32) # (N, 2)
        
        depot_xy_np = (depot_raw - self.min_coord) / self.scale_coord
        node_xy_np = (node_raw - self.min_coord) / self.scale_coord
        
        # Chuyển thành Tensor và thêm Batch Dimension
        # FIX QUAN TRỌNG: Cần ensure depot_xy có shape (1, 1, 2)
        # depot_xy_np shape là (2,). 
        # Cần biến đổi thành (1, 2) rồi unsqueeze -> (1, 1, 2)
        
        depot_tensor_raw = torch.tensor(depot_xy_np, dtype=torch.float32, device=self.device)
        if depot_tensor_raw.ndim == 1:
            depot_tensor_raw = depot_tensor_raw.unsqueeze(0) # (1, 2)
            
        self.depot_xy = depot_tensor_raw.unsqueeze(0) # (1, 1, 2)
        self.node_xy = torch.tensor(node_xy_np, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, N, 2)
        
        # 2. Normalize Demand (1, N)
        raw_demands = customers['Demand'].values.astype(np.float32)
        self.node_demand = torch.tensor(raw_demands / cfg.VEHICLE_CAPACITY, 
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 3. Normalize Time
        # Depot TW cần (1, 1, 2)
        d_tw = np.array([[depot['Start_Time'], depot['End_Time']]]).astype(np.float32) # (1, 2)
        self.depot_tw = torch.tensor(d_tw / self.max_time, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        c_tw = customers[['Start_Time', 'End_Time']].values.astype(np.float32)
        self.time_windows = torch.tensor(c_tw / self.max_time, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        serv = customers['ServiceTime'].values.astype(np.float32)
        self.service_times = torch.tensor(serv / self.max_time, dtype=torch.float32, device=self.device).unsqueeze(0)

#
#
#
# #
# #
# #
# #
# ##
    def _prepare_pomo_tensors_tự_fix_thêm_unsqueeze(self, df):
        """Normalize dữ liệu Static sang khoảng [0, 1]"""
        depot = df[df['Number'] == 0].iloc[0]
        customers = df[df['Number'] != 0]
        
        # 1. Normalize Coordinates (Lat/Lon -> 0..1)
        depot_coords = depot[['Longitude', 'Latitude']].values.astype(np.float32)
        node_coords = customers[['Longitude', 'Latitude']].values.astype(np.float32)
        
        depot_xy_np = (depot_coords - self.min_coord) / self.scale_coord
        node_xy_np = (node_coords - self.min_coord) / self.scale_coord
        
        self.depot_xy = torch.tensor(depot_xy_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.node_xy = torch.tensor(node_xy_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 2. Normalize Demand
        raw_demands = customers['Demand'].values.astype(np.float32)
        self.node_demand = torch.tensor(raw_demands / cfg.VEHICLE_CAPACITY, 
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 3. Normalize Time
        d_tw = np.array([[depot['Start_Time'], depot['End_Time']]]).astype(np.float32)
        self.depot_tw = torch.tensor(d_tw / self.max_time, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        c_tw = customers[['Start_Time', 'End_Time']].values.astype(np.float32)
        self.time_windows = torch.tensor(c_tw / self.max_time, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        serv = customers['ServiceTime'].values.astype(np.float32)
        self.service_times = torch.tensor(serv / self.max_time, dtype=torch.float32, device=self.device).unsqueeze(0)

    def generate_candidates(self, dual_values_dict):
        # --- Prepare Dynamic Context (Duals) ---
        # Note: Index iteration theo locations_df
        # Get duals tương ứng với từng customer theo thứ tự trong self.node_xy
        
        duals_list = []
        for idx in range(len(self.locations_df)):
            row = self.locations_df.iloc[idx]
            if int(row['Number']) == 0: continue # Skip Depot Dual
            d_val = dual_values_dict.get(int(row['Number']), 0.0)
            duals_list.append(d_val)
            
        duals_tensor = torch.tensor(duals_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        norm_duals = duals_tensor / self.max_time
        
        prices_dummy = torch.zeros_like(self.travel_times_tensor)

        input_dict = {
            'problem_size': self.problem_size, # <--- KEY 'problem_size'
            'depot_xy': self.depot_xy,
            'node_xy': self.node_xy,
            'node_demand': self.node_demand,
            'time_windows': self.time_windows,
            'depot_time_window': self.depot_tw,
            'service_times': self.service_times,
            'duals': norm_duals, 
            'travel_times': self.travel_times_tensor,
            'prices': prices_dummy 
        }

        # --- POMO Inference ---
        try:
            # Model forward trả về node index sequence
            raw_routes_idx = self.pomo_gen(input_dict)
            raw_routes_np = raw_routes_idx.squeeze(0).cpu().numpy()
        except Exception as e:
            # Print full stacktrace để debug
            import traceback
            traceback.print_exc()
            print(f"[Pricing Error] POMO Inference failed: {e}")
            return []

        candidates_set = set()
        '''
        for i in range(raw_routes_np.shape[0]):
            route_ids = raw_routes_np[i]
            clean_route = self._parse_pomo_output(route_ids)
            if clean_route:
                candidates_set.add(tuple(clean_route))
        '''
        # --- DEBUG BLOCK START ---
        print(f"\n[DEBUG] Raw output from POMO (first 3 samples): {raw_routes_np[:3]}")
        
        valid_syntax_count = 0
        valid_feasible_count = 0
        negative_rc_count = 0
        
        candidates_set = set()
        for i in range(raw_routes_np.shape[0]):
            route_ids = raw_routes_np[i]
            clean_route = self._parse_pomo_output(route_ids) # Hàm này check cú pháp start-end
            
            if clean_route:
                valid_syntax_count += 1
                
                # Check Feasibility sơ bộ (Ví dụ Capacity)
                load = sum([self.locations_df.iloc[n]['Demand'] for n in clean_route])
                if load <= cfg.VEHICLE_CAPACITY: # Bạn có thể thêm check Time ở đây
                    valid_feasible_count += 1
                    
                    # Check Reduced Cost thủ công để log
                    rc_val = self._calc_debug_rc(clean_route, dual_values_dict)
                    if rc_val < -1e-4:
                        negative_rc_count += 1
                        candidates_set.add(tuple(clean_route))
                    else:
                        pass 
                        # print(f"  > Route feasible but positive RC: {rc_val:.2f}")
        
        print(f"[DEBUG Stats] Syntax OK: {valid_syntax_count}, Feasible: {valid_feasible_count}, Negative RC: {negative_rc_count}")
        # --- DEBUG BLOCK END ---
        
        final_candidates = list(candidates_set)

        # --- FALLBACK ---
        if len(final_candidates) == 0:
            print("[Pricing] Heuristic yielded 0 candidates. Using Exact Fallback.")
            exact_routes = self.exact_solver.solve(dual_values_dict)
            for r in exact_routes:
                final_candidates.append(tuple(r))
                
        return [list(r) for r in final_candidates]

    def _parse_pomo_output(self, node_indices):
        route = [0]
        has_customer = False
        real_ids = self.locations_df['Number'].values # Array full including depot [0]
        
        # Note: Input static đã filter depot ra ngoài khi tạo self.node_xy
        # Do đó index i của POMO (từ 1..N) ứng với customer thứ i-1 trong list customers
        # List customer: locations_df[1:]
        
        customer_real_ids = self.locations_df.iloc[1:]['Number'].values
        
        for node_idx in node_indices:
            idx = int(node_idx)
            if idx == 0:
                if has_customer:
                    route.append(0)
                    break # Kết thúc route
            else:
                # idx là 1-based index của node.
                # cần map sang index 0-based của mảng customer
                if idx - 1 < len(customer_real_ids):
                    real_id = customer_real_ids[idx - 1]
                    route.append(int(real_id))
                    has_customer = True
        
        if len(route) > 1 and route[-1] != 0: route.append(0)
        
        if len(route) > 2: return route
        return None