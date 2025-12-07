import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from torch_geometric.loader import DataLoader # Batching for Graphs

from agents.gnn_selector import BipartiteGNN
from data.bipartite_builder import BipartiteGraphBuilder
from utils.utils import save_checkpoint, get_checkpoint_path
from configs import main_config as cfg


class DQNAgent:
    """
    RL Selector quản lý việc Training và Action Selection.
    Dùng Double DQN logic để ổn định.
    """
    def __init__(self):
        self.device = cfg.DEVICE
        
        # Builder để biến data thô thành Tensor khi chạy step()
        self.graph_builder = BipartiteGraphBuilder(self.device)

        # 1. Networks
        self.policy_net = BipartiteGNN().to(self.device)
        self.target_net = BipartiteGNN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 2. Optimizer & Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.DQN_LR)
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss ổn định hơn MSE

        # 3. Replay Buffer (Lưu trữ list các HeteroData object)
        self.memory = deque(maxlen=cfg.MEMORY_CAPACITY)
        
        # 4. Parameters
        self.epsilon = cfg.DQN_EPSILON_START
        self.steps_done = 0

        # Logger placeholder (gắn sau)
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def select_action(self, obs_dict, training=True):
        """
        Chọn candidate tốt nhất để add vào RMP.
        Input: obs_dict (numpy raw data từ Env)
        Return: int (index trong danh sách candidates) hoặc None nếu không có candidates
        """
        # List candidate gốc (index trong hệ thống)
        candidate_indices = obs_dict.get('candidates_indices', [])
        if len(candidate_indices) == 0:
            return None # Không có action khả thi

        # Epsilon-Greedy Logic
        if training and random.random() < self.epsilon:
            return random.choice(range(len(candidate_indices)))

        # Inference mode
        # 1. Convert to PyG Graph
        data = self.graph_builder.transform(obs_dict)
        
        # 2. Forward Pass
        self.policy_net.eval()
        with torch.no_grad():
            q_values_all = self.policy_net(data.x_dict, data.edge_index_dict)
            # q_values_all có shape (Total_Columns_in_RMP_plus_candidates, )
            
            # 3. Masking: Chỉ lấy Q-value của Candidates
            # mask lấy từ builder
            mask = data['column'].candidate_mask 
            candidate_q_values = q_values_all[mask] # Lấy giá trị ứng với True
            
            # candidate_q_values sẽ có độ dài = len(candidate_indices)
            # Lấy argmax trong đám candidate
            best_idx_local = candidate_q_values.argmax().item()
            
            return best_idx_local

    def store_transition(self, obs, action_idx, reward, next_obs, done):
        """Lưu trải nghiệm vào Replay Buffer"""
        state_data = self.graph_builder.transform(obs)
        
        # Action ở đây là index trong mảng candidates
        # Ta cần map nó về global index nếu cần, nhưng để đơn giản DQN chỉ cần local index của candidates
        # Nhưng việc re-construct lại candidate q-values từ buffer hơi phức tạp với PyG batching.
        # ==> STRATEGY: Lưu cả obs_dict raw, khi train mới convert batch. 
        # (Cách này chậm hơn lúc train nhưng tiết kiệm VRAM/RAM hơn cho Buffer Graph)
        
        self.memory.append((obs, action_idx, reward, next_obs, done))

    def train_step(self):
        """Lấy 1 batch từ buffer và update mạng"""
        if len(self.memory) < cfg.BATCH_SIZE:
            return None

        # 1. Sample Batch raw
        batch_raw = random.sample(self.memory, cfg.BATCH_SIZE)
        
        # 2. Process Batch to PyG Graphs
        data_list_curr = []
        data_list_next = []
        actions = []
        rewards = []
        dones = []

        valid_batch_count = 0
        for (obs, action_idx, reward, next_obs, done) in batch_raw:
            if action_idx is None: 
                continue # Skip bad transitions
                
            g_curr = self.graph_builder.transform(obs)
            # Kiểm tra xem graph build ra có khớp số lượng candidate không
            # obs['candidates_indices'] có size X, action phải < X
            num_cands = len(obs.get('candidates_indices', []))
            if action >= num_cands:
                continue # Skip invalid index
                
            g_curr.taken_action_local_idx = action 
            data_list_curr.append(g_curr)

            g_next = self.graph_builder.transform(next_obs)
            data_list_next.append(g_next)

            actions.append(action)
            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            valid_batch_count += 1
            
        if valid_batch_count == 0:
            return None # Không có sample nào hợp lệ trong batch random này

        # DataLoader của PyG tự động ghép các graph con thành 1 graph lớn (Batch Graph)
        # Các nodes được đánh số lại liên tục. 
        # Cần xử lý action masking cực kỳ cẩn thận ở đây.
        
        batch_curr = next(iter(DataLoader(data_list_curr, batch_size=cfg.BATCH_SIZE, shuffle=False)))
        batch_next = next(iter(DataLoader(data_list_next, batch_size=cfg.BATCH_SIZE, shuffle=False)))
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 3. Calculate Current Q (Policy Net)
        self.policy_net.train()
        
        # Forward pass batch lớn
        # output shape: (Sum_Nodes_In_Batch, )
        q_values_curr_all = self.policy_net(batch_curr.x_dict, batch_curr.edge_index_dict)
        
        # --- EXTRACT CHOSEN ACTION Q-VALUES ---
        
        # 1. Lọc lấy các node là candidate
        cand_mask_curr = batch_curr['column'].candidate_mask
        q_cand_only = q_values_curr_all[cand_mask_curr] 
        # q_cand_only là vector nối đuôi: [G1_cands, G2_cands, ...]
        
        # Ta cần tính cumulative count số candidates mỗi graph để shift index
        num_cands_per_graph = [d['column'].candidate_mask.sum().item() for d in data_list_curr]
        
        # Tạo index cho action đã chọn trong vector q_cand_only
        start_indices = np.cumsum([0] + num_cands_per_graph[:-1])
        global_action_indices = start_indices + np.array(actions)
        
        # Gather Q values đã chọn
        # Chuyển numpy -> tensor long
        global_action_indices = torch.tensor(global_action_indices, dtype=torch.long, device=self.device)
        
        pred_q_values = q_cand_only[global_action_indices]

        # 4. Calculate Target Q (Target Net) with Double DQN
        with torch.no_grad():
            q_values_next_all = self.target_net(batch_next.x_dict, batch_next.edge_index_dict)
            
            # Tương tự, lấy Max Q trong số các candidates của trạng thái tiếp theo
            cand_mask_next = batch_next['column'].candidate_mask
            q_cand_next = q_values_next_all[cand_mask_next]
            
            # Cần split q_cand_next ra từng graph để lấy max
            num_cands_next = [d['column'].candidate_mask.sum().item() for d in data_list_next]
            
            # Đoạn này xử lý Max Pooling cho ragged tensor thủ công
            # (Hoặc dùng torch_scatter.scatter_max nếu cài thêm lib, ở đây dùng split_with_sizes thuần Torch)
            # Có thể có graph không có candidate nào (dừng sớm), num_cands = 0
            
            # Handle trường hợp num_cands_next có phần tử 0 -> split bị lỗi hoặc empty tensor
            # Fallback logic: next_q = 0 nếu done hoặc không có candidate
            
            q_next_split = torch.split(q_cand_next, num_cands_next)
            max_next_q = []
            for q_vals in q_next_split:
                if len(q_vals) > 0:
                    max_next_q.append(q_vals.max())
                else:
                    # Không có candidates ở next step
                    max_next_q.append(torch.tensor(0.0, device=self.device))
            
            max_next_q = torch.stack(max_next_q)
            
            expected_q = rewards + (1 - dones) * cfg.DQN_GAMMA * max_next_q

        # 5. Backward
        loss = self.loss_fn(pred_q_values, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 6. Update Target & Epsilon
        self.steps_done += 1
        self.update_epsilon()
        
        if self.steps_done % cfg.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def update_epsilon(self):
        decay = (cfg.DQN_EPSILON_START - cfg.DQN_EPSILON_END) / cfg.DQN_EPSILON_DECAY
        self.epsilon = max(cfg.DQN_EPSILON_END, self.epsilon - decay)

    def save(self, path=None):
        """Lưu model"""
        # Nếu path truyền vào None, tự động tạo theo config
        if path is None:
            path = get_checkpoint_path(cfg.DIR_DQN_OUTPUT, "dqn_agent", self.steps_done)
            
        state = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }
        torch.save(state, path)
        if self.logger:
            self.logger.info(f"Model saved to {path}")
            
    def load(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epsilon = ckpt.get('epsilon', self.epsilon)
            self.steps_done = ckpt.get('steps_done', 0)
            if self.logger:
                self.logger.info(f"Loaded DQN from {path} (Step {self.steps_done})")