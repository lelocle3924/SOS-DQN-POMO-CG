import torch
import torch.nn as nn
from .pomo_support.attention_model import AttentionModel
from .pomo_support.pomo_gym_env import PomoGymEnv
from configs import main_config as cfg
import os

class POMOGenerator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Params khớp config train
        self.model_params = {
            'embedding_dim': cfg.POMO_EMBEDDING_DIM,
            'sqrt_embedding_dim': cfg.POMO_EMBEDDING_DIM**(1/2),
            'encoder_layer_num': cfg.POMO_ENCODER_LAYER_NUM,
            'qkv_dim': cfg.POMO_QKV_DIM,
            'head_num': cfg.POMO_HEAD_NUM,
            'logit_clipping': cfg.POMO_LOGIT_CLIPPING,
            'ff_hidden_dim': cfg.POMO_FF_HIDDEN_DIM,
            'eval_type': cfg.POMO_EVAL_TYPE,
        }
        
        self.model = AttentionModel(**self.model_params)
        self.model.to(device)
        self.model.eval()
        
        self._load_checkpoint()
        self.env = None

    def _load_checkpoint(self):
        if not os.path.exists(cfg.POMO_MODEL_PATH):
            print(f"[POMO] Checkpoint not found: {cfg.POMO_MODEL_PATH}. Using RANDOM weights.")
            return

        print(f"[POMO] Loading weights from {cfg.POMO_MODEL_PATH}...")
        try:
            map_loc = self.device if torch.cuda.is_available() else 'cpu'
            # Dùng False cho local trusted checkpoint
            checkpoint = torch.load(cfg.POMO_MODEL_PATH, map_location=map_loc, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"[POMO] Error loading weights: {e}")

    def forward(self, input_data):
        """Rollout"""
        p_size = input_data['problem_size']
        if self.env is None or self.env.problem_size != p_size:
            self.env = PomoGymEnv(problem_size=p_size, pomo_size=p_size)
            
        # PASS TRỰC TIẾP, nhờ **kwargs bên Env xử lý phần thừa
        self.env.load_specific_problem(**input_data)
        
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            
            state, _, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, _, done = self.env.step(selected)
                
        return self.env.selected_node_list