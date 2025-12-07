import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

from configs import main_config as cfg

class BipartiteGNN(nn.Module):
    """
    Neural Network chọn cột (DQN Agent) dùng kiến trúc Message Passing.
    
    Flow:
    1. Embedding layers cho Col features và Con features.
    2. Message Passing: 
       - Col -> Con (Aggregation)
       - Con -> Col (Update features cột dựa trên Duals của Constraints)
    3. Q-Head: Dự đoán Q-value cho từng Column node.
    """
    def __init__(self):
        super(BipartiteGNN, self).__init__()
        
        self.hidden_dim = cfg.GNN_HIDDEN_DIM
        
        # --- 1. Init Projectors (Embedding Inputs) ---
        self.lin_col = Linear(cfg.GNN_COL_FEAT_DIM, self.hidden_dim)
        self.lin_con = Linear(cfg.GNN_CON_FEAT_DIM, self.hidden_dim)

        # --- 2. Message Passing Layers ---
        # HeteroConv cho phép định nghĩa conv riêng cho từng loại cạnh
        # Layer 1
        self.conv1 = HeteroConv({
            ('column', 'serves', 'constraint'): SAGEConv((self.hidden_dim, self.hidden_dim), self.hidden_dim),
            ('constraint', 'served_by', 'column'): SAGEConv((self.hidden_dim, self.hidden_dim), self.hidden_dim),
        }, aggr='mean')

        # Layer 2 (Có thể thêm nếu muốn deep hơn)
        self.conv2 = HeteroConv({
            ('column', 'serves', 'constraint'): SAGEConv(self.hidden_dim, self.hidden_dim),
            ('constraint', 'served_by', 'column'): SAGEConv(self.hidden_dim, self.hidden_dim),
        }, aggr='mean')

        # --- 3. Q-Value Predictor Head ---
        # Input là embedding cuối cùng của node 'column' -> Scalar Q-value
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1) # Output 1 giá trị Q
        )

        self.to(cfg.DEVICE)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass xử lý cả đồ thị (batch hoặc single).
        x_dict: Dictionary chứa features của {'column': x_c, 'constraint': x_n}
        edge_index_dict: Dictionary chứa indices các loại cạnh.
        """
        
        # 1. Project Input Features
        x_dict['column'] = F.relu(self.lin_col(x_dict['column']))
        x_dict['constraint'] = F.relu(self.lin_con(x_dict['constraint']))

        # 2. Graph Convolutions
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()} # Non-linearity
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 3. Predict Q-Values
        # Ta chỉ quan tâm embedding của cột ('column')
        col_embeddings = x_dict['column'] 
        q_values = self.q_head(col_embeddings) # (Total_Cols, 1)
        
        return q_values.squeeze(-1) # (Total_Cols, )