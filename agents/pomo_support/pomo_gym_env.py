from dataclasses import dataclass
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    time_windows: torch.Tensor = None
    # shape: (batch, problem, 2)
    duals: torch.Tensor = None
    # shape: (batch, problem)
    service_times: torch.Tensor = None
    travel_times: torch.Tensor = None
    prices: torch.Tensor = None
    depot_time_window: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    current_times: torch.Tensor = None
    current_prices: torch.Tensor = None

class PomoGymEnv(gym.Env):
    def __init__(self, problem_size, pomo_size, **env_params):
        super().__init__()
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_specific_problem(self, depot_xy, node_xy, node_demand, 
                              time_windows, depot_time_window, 
                              service_times, duals, 
                              travel_times, prices,
                              **kwargs):
        """
        Load dữ liệu cụ thể vào Env. 
        **kwargs: Sẽ nuốt 'problem_size' hoặc bất kỳ key thừa nào từ input dict
        """
        self.batch_size = 1 
        
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        
        depot_demand = torch.zeros((self.batch_size, 1), device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        
        depot_service = torch.zeros((self.batch_size, 1), device=service_times.device)
        self.depot_node_service_time = torch.cat((depot_service, service_times), dim=1)
        
        self.depot_node_time_windows = torch.cat((depot_time_window, time_windows), dim=1)
        
        # Duals cũng cần padding depot
        # Fix logic cũ: depot_dual tạo manual, node duals truyền vào
        # Check if depot duals exists, if not create zeros
        depot_dual_pad = torch.zeros((self.batch_size, 1), device=duals.device)
        self.depot_node_duals = torch.cat((depot_dual_pad, duals), dim=1)
        
        self.prices = prices
        self.travel_times = travel_times 

        self.BATCH_IDX = torch.arange(self.batch_size, device=node_xy.device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=node_xy.device)[None, :].expand(self.batch_size, self.pomo_size)

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.time_windows = time_windows
        self.reset_state.depot_time_window = depot_time_window
        self.reset_state.duals = duals
        self.reset_state.service_times = service_times
        self.reset_state.prices = prices
        self.reset_state.travel_times = travel_times

        print(f"\n[ENV DATA CHECK] Load Problem:")
        print(f"  > Coords shape: {self.depot_node_xy.shape} | Range: [{self.depot_node_xy.min():.3f}, {self.depot_node_xy.max():.3f}]")
        print(f"  > TimeWindows shape: {self.depot_node_time_windows.shape}")
        # Quan trọng: kiểm tra start/end windows. Nếu end < start hoặc scale sai -> mask sẽ sai.
        tw_sample = self.depot_node_time_windows[0, 1, :]
        print(f"  > TW Sample (Node 1): {tw_sample.tolist()} (Nên là [start, end] trong [0,1])")
        print(f"  > Duals range: [{self.depot_node_duals.min():.4f}, {self.depot_node_duals.max():.4f}]")
        print("------------------------------------------")

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        
        self.current_times = torch.zeros((self.batch_size, self.pomo_size), device=self.BATCH_IDX.device)
        self.current_prices = torch.zeros((self.batch_size, self.pomo_size), device=self.BATCH_IDX.device)
        
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.BATCH_IDX.device)
        
        self.at_the_depot = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.bool, device=self.BATCH_IDX.device)
        self.load = torch.ones((self.batch_size, self.pomo_size), device=self.BATCH_IDX.device)
        
        self.visited_ninf_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), device=self.BATCH_IDX.device)
        self.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), device=self.BATCH_IDX.device)
        self.finished = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.bool, device=self.BATCH_IDX.device)
        
        self.step_state.finished = self.finished
        
        return self.reset_state, None, False

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_times = self.current_times
        self.step_state.current_prices = self.current_prices
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        if self.selected_count == 0: # Bước khởi đầu
            # Kiểm tra mask. Nếu mask toàn -inf ngoại trừ depot -> Lỗi Logic Env
            # mask shape: (batch, pomo, num_nodes+1)
            # Ta lấy sample đầu tiên để soi
            sample_mask = self.ninf_mask[0, 0, :10] # 10 node đầu
            print(f"[ENV DEBUG] Initial Mask (First 10 nodes): {sample_mask}")
            
            # Check Demands input
            print(f"[ENV DEBUG] Node Demands (First 5): {self.depot_node_demand[0, 1:6]}")
            print(f"[ENV DEBUG] Depot Capacity Load: {self.load[0, 0]}") # Should be 1.0
        
        return self.step_state, None, False
    
    def step(self, selected):
        self.selected_count += 1
        
        if self.current_node is None:
            previous_node = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.long, device=selected.device)
        else:
            previous_node = self.current_node
            
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        if self.selected_count > 1:
            self.at_the_depot = (selected == 0)

        batch_idx = self.BATCH_IDX
        pomo_idx = self.POMO_IDX
        prev_idx = previous_node
        curr_idx = selected

        selected_demand = self.depot_node_demand[batch_idx, curr_idx]
        self.load -= selected_demand 

        travel_time_val = self.travel_times[batch_idx, prev_idx, curr_idx]
        service_time_val = self.depot_node_service_time[batch_idx, curr_idx]
        tw_start_val = self.depot_node_time_windows[batch_idx, curr_idx, 0]

        arrival_time = self.current_times + travel_time_val
        start_service = torch.maximum(arrival_time, tw_start_val)
        self.current_times = start_service + service_time_val
        
        price_val = self.prices[batch_idx, prev_idx, curr_idx]
        self.current_prices += price_val

        self.visited_ninf_flag[batch_idx, pomo_idx, selected] = float('-inf')
        
        if self.selected_count == 2: 
             self.visited_ninf_flag[:, :, 0] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        
        round_error = 1e-5
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        demand_too_large = (self.load[:, :, None] + round_error) < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        # Gather Logic
        curr_node_expanded = self.current_node[:, :, None, None].expand(-1, -1, 1, self.problem_size + 1)
        travel_matrix_from_curr = self.travel_times.unsqueeze(1).expand(-1, self.pomo_size, -1, -1) 
        
        travel_times_to_next = torch.gather(
            travel_matrix_from_curr, 
            2, 
            curr_node_expanded
        ).squeeze(2)
        
        future_arrivals = self.current_times[:, :, None] + travel_times_to_next
        tw_end_list = self.depot_node_time_windows[:, :, 1].unsqueeze(1).expand(-1, self.pomo_size, -1)
        
        too_late = (future_arrivals > tw_end_list)
        self.ninf_mask[too_late] = float('-inf')

        newly_finished = self.at_the_depot
        self.finished = self.finished | newly_finished
        
        # Mask toàn bộ nếu đã finish
        self.ninf_mask[self.finished] = float('-inf')
        
        self.ninf_mask[:, :, 0][self.finished] = 0 

        done = self.finished.all()
        
        return self.step_state, self.current_prices, done