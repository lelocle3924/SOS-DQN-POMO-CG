import gymnasium as gym
import numpy as np
import torch
import logging

from envs.rmp_service import RMPService
from solvers.heuristic_pricing import PomoPricingAgent
from data.processor import process_instance_data
from configs import main_config as cfg
from utils.utils import parse_args

args = parse_args()
''' 
OWNS
- RMP Service
- Pricing Agent
- State Tracking
'''
class VRPCGEnv(gym.Env):
    """
    Môi trường Gym chuẩn cho bài toán chọn cột (Column Selection).
    """
    def __init__(self):
        super(VRPCGEnv, self).__init__()
        self.logger = logging.getLogger('root')
        
        # 1. Load & Process Static Data
        # Dữ liệu tĩnh load 1 lần dùng mãi mãi
        if args.mode == 'train':
            INSTANCE_FILE = cfg.TRAIN_INSTANCE_FILE
        else:
            INSTANCE_FILE = cfg.TEST_INSTANCE_FILE

        self.logger.info(f"Env: Loading data from {INSTANCE_FILE}")
        self.locations_df, _, self.time_matrix = process_instance_data(INSTANCE_FILE)        
        # Tensorize cho POMO
        self.dist_tensor = torch.tensor(self.time_matrix, dtype=torch.float32, device=cfg.DEVICE)
        
        # 2. Init Sub-modules
        # RMP Service sẽ init mỗi lần reset()
        self.rmp_service = None
        
        # Pricing Agent (POMO Wrapper)
        self.pricing_solver = PomoPricingAgent(self.locations_df, self.dist_tensor)
        
        # 3. State tracking
        self.current_candidates = [] # List các routes (candidates) hiện tại
        self.init_obj = 0.0
        self.prev_obj = 0.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Init RMP
        customers = self.locations_df[self.locations_df['Number'] != 0]
        self.rmp_service = RMPService(
            customers_df=customers,
            vehicle_capacity=cfg.VEHICLE_CAPACITY,
            big_m=cfg.RMP_BIG_M
        )
        
        # Initial Columns (Dummy hoặc Heuristic)
        # Để code gọn, ta dùng dummy 1-1 ở đây (Production nên dùng NN)
        init_routes = []
        init_costs = []
        customer_ids = [int(r['Number']) for _, r in customers.iterrows()]
        
        for cid in customer_ids:
            route = [0, cid, 0]
            # Lấy index trong ma trận (đã sort theo Number)
            # Vì Number 0 là index 0, nên Number k là index k (nếu data chuẩn)
            idx = self.locations_df.index.get_loc(cid) 
            idx_0 = 0 
            cost = self.time_matrix[idx_0, idx] + self.time_matrix[idx, idx_0] + cfg.DEFAULT_SERVICE_TIME
            
            init_routes.append(route)
            init_costs.append(cost)
            
        self.rmp_service.add_initial_columns(init_routes, init_costs)
        
        # First Solve
        obj, duals, _ = self.rmp_service.solve()
        if obj is None:
            raise RuntimeError("Initial RMP infeasible.")
            
        self.init_obj = obj
        self.prev_obj = obj
        
        # Generate first candidates
        self._run_pricing(duals)
        
        return self._get_observation(duals)
        
    def step(self, action_idx):
        """
        Action idx: Index trong list self.current_candidates
        """
        # Nếu Agent chọn invalid action hoặc không có candidate
        if action_idx is None or action_idx >= len(self.current_candidates):
            self.logger.warning("Agent selected invalid action or no candidates.")
            return self._get_dummy_observation(), 0, True, False, {}

        # 1. Execute Action: Add column to RMP
        selected_route = self.current_candidates[action_idx]
        
        # Tính cost thật
        cost = self._calculate_route_cost(selected_route)
        self.rmp_service.add_column(selected_route, cost)
        
        # 2. Resolve RMP
        new_obj, duals, _ = self.rmp_service.solve()
        
        # 3. Calculate Reward
        terminated = False
        reward = 0
        
        if new_obj is None:
            # Lỗi solver
            reward = -1.0
            terminated = True
        else:
            # Reward là phần cải thiện so với obj ban đầu (để scale reward to)
            # REWARD REWARD REWARD
            reward = (self.prev_obj - new_obj) / self.init_obj * 100.0
            self.prev_obj = new_obj
            
            # 4. Generate New Candidates (Next State Logic)
            self._run_pricing(duals)
            
            if len(self.current_candidates) == 0:
                self.logger.info("Environment Converged (No negative columns).")
                terminated = True
        
        obs = self._get_observation(duals) if not terminated else self._get_dummy_observation()
        
        return obs, reward, terminated, False, {"obj": new_obj}

    def _run_pricing(self, duals):
        """Chạy POMO để tìm cột âm"""
        raw_candidates = self.pricing_solver.generate_candidates(duals)
        
        # Lọc lại reduced cost < 0 thật kỹ
        self.current_candidates = []
        for r in raw_candidates:
            rc = self._calculate_reduced_cost(r, duals)
            if rc < -1e-4:
                self.current_candidates.append(r)
                
    def _get_observation(self, duals):
        """Xây dựng dict features cho GNN Builder"""
        customer_ids = self.rmp_service.customer_ids
        id_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
        
        # 1. Con Feats
        con_features = [[duals.get(cid, 0.0), 1.0] for cid in customer_ids]
        
        # 2. Col Feats
        col_features = []
        edges_src, edges_dst = [], []
        cand_indices = []
        
        # Routes đang có trong RMP
        existing_cols_cnt = len(self.rmp_service.routes_data)
        all_routes = self.rmp_service.routes_data + self.current_candidates
        
        '''
        KÉM HIỆU QUẢ KHI SỐ COLUMNS LÊN CAO
        '''
        for idx, route in enumerate(all_routes):
            #tìm ra cái nào là candidate
            is_cand = 1.0 if idx >= existing_cols_cnt else 0.0
            if is_cand: cand_indices.append(idx)
            
            #tính reduced cost
            rc = self._calculate_reduced_cost(route, duals)
            #nếu là route có trước thì lấy nguyên cost, nếu là candidate mới thêm thì gọi hàm tính route cost
            real_cost = self.rmp_service.column_costs[idx] if idx < existing_cols_cnt else self._calculate_route_cost(route)
            
            # Feat: [RC, Cost, Length, IsCand]
            col_features.append([rc, real_cost/1000.0, len(route)/10.0, is_cand, 0,0,0,0])
            
            for node in route:
                if node in id_to_idx:
                    edges_src.append(idx)
                    edges_dst.append(id_to_idx[node])
                    
        return {
            'col_features': np.array(col_features, dtype=np.float32),
            'con_features': np.array(con_features, dtype=np.float32),
            'edge_indices': np.array([edges_src, edges_dst]),
            'candidates_indices': cand_indices,
            'candidates_routes': self.current_candidates
        }

    def _calculate_route_cost(self, route):
        c = 0.0
        for i in range(len(route)-1):
            c += self.time_matrix[route[i], route[i+1]]
        return c

    def _calculate_reduced_cost(self, route, duals):
        cost = self._calculate_route_cost(route)
        dual_sum = sum([duals.get(n, 0.0) for n in route[1:-1]]) # Skip depots
        return cost - dual_sum

    def _get_dummy_observation(self):
        # Trả về obs rỗng để DQN không crash khi Done
        return {
            'col_features': np.zeros((1, 8)),
            'con_features': np.zeros((1, 2)),
            'edge_indices': np.zeros((2, 0)),
            'candidates_indices': []
        }