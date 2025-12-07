import torch
import numpy as np
from torch_geometric.data import HeteroData
from configs import main_config as cfg

class BipartiteGraphBuilder:
    """
    Chuyển đổi trạng thái Environment thành PyG HeteroData Object.
    Graph Type: Bipartite
    - Nodes Type 'constraint': Khách hàng/Ràng buộc
    - Nodes Type 'column': Các routes hiện có + Candidate routes mới sinh
    """
    
    def __init__(self, device):
        self.device = device

    def transform(self, observation_dict):
        """
        Input: dict chứa các numpy arrays thô (đã xử lý từ env):
            - col_features: (Num_Cols, 8)
            - con_features: (Num_Cons, 2)
            - edge_indices: (2, Num_Edges) [[col_idx...], [con_idx...]]
            - candidates_indices: List[int] - chỉ mục của các cột là ứng viên (candidates)
        Output: PyG HeteroData Object
        """
        data = HeteroData()

        # 1. Node Features (Chuyển sang Tensor & move to Device)
        # Type: Constraint (Left nodes)
        con_x = torch.tensor(observation_dict['con_features'], dtype=torch.float32, device=self.device)
        data['constraint'].x = con_x
        data['constraint'].num_nodes = con_x.shape[0]

        # Type: Column (Right nodes - Bao gồm cả candidates)
        col_x = torch.tensor(observation_dict['col_features'], dtype=torch.float32, device=self.device)
        data['column'].x = col_x
        data['column'].num_nodes = col_x.shape[0]

        # 2. Edges (Connectivity)
        # PyG yêu cầu edge_index dạng long (int64)
        # Trong Bipartite: Col -> Con (Route phục vụ khách hàng)
        edge_index = torch.tensor(observation_dict['edge_indices'], dtype=torch.long, device=self.device)
        
        # Define hướng 1: Column -> Constraint
        data['column', 'serves', 'constraint'].edge_index = edge_index

        # Define hướng 2: Constraint -> Column (Message passing chiều ngược lại)
        # Flip rows của edge_index: [0] là Col, [1] là Con -> Flip thành [1] Con, [0] Col
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        data['constraint', 'served_by', 'column'].edge_index = reverse_edge_index

        # 3. Action Masking (Candidate Info)
        # Chúng ta cần biết những Node nào trong 'column' là candidates để predict Q-value
        # Candidates indices mapping: index trong col_x
        cand_indices = torch.tensor(observation_dict['candidates_indices'], dtype=torch.long, device=self.device)
        
        # Lưu mask để model biết chỉ predict cho những node này
        # Tạo mask bool kích thước bằng số lượng columns
        candidate_mask = torch.zeros(col_x.shape[0], dtype=torch.bool, device=self.device)
        if len(cand_indices) > 0:
            candidate_mask[cand_indices] = True
            
        data['column'].candidate_mask = candidate_mask
        
        # Metadata khác để hỗ trợ Replay Buffer
        # (Số lượng node thay đổi mỗi step nên cần batch vector nếu dùng DataLoader)
        
        return data

# Helper dummy function để test builder (nếu chạy độc lập)
if __name__ == "__main__":
    dummy_obs = {
        'col_features': np.random.rand(10, 8),
        'con_features': np.random.rand(5, 2),
        'edge_indices': np.array([[0, 0, 1, 2], [0, 1, 1, 3]]), # Col 0 -> Con 0, 1...
        'candidates_indices': [8, 9] # Cột 8, 9 là hàng mới thêm
    }
    builder = BipartiteGraphBuilder(torch.device('cpu'))
    out = builder.transform(dummy_obs)
    print("Graph built:", out)
    print("Edges type 1:", out['column', 'serves', 'constraint'].edge_index.shape)