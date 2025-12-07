import os
import torch
from datetime import datetime

'''
--- PATHS ---
'''
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'inputs')
RESULTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Auto create dirs
DIR_DQN_OUTPUT = os.path.join(RESULTS_DIR, 'dqn_model')
DIR_POMO_OUTPUT = os.path.join(RESULTS_DIR, 'pretrained_pomo')
DIR_VRP_RESULTS = os.path.join(RESULTS_DIR, 'vrp_results')

for d in [RESULTS_DIR, DIR_DQN_OUTPUT, DIR_POMO_OUTPUT, DIR_VRP_RESULTS]:
    os.makedirs(d, exist_ok=True)

# Training Log
current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
LOG_FILE_TRAIN = os.path.join(RESULTS_DIR, f'train_log_{current_time_str}.txt')

'''
# --- PROBLEM SETTINGS ---
'''
VEHICLE_CAPACITY = 1000.0
VEHICLE_SPEED_KMH = 40.0
DEFAULT_SERVICE_TIME = 10.0
DEFAULT_DEMAND = 1.0

'''
# --- SYSTEM ---
'''
USE_CUDA = True  
if USE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

'''
# --- CG PARAMETERS ---
'''
RMP_BIG_M = 1e6
# Column Management
MAX_COLS_PER_ITER = 20 # Số cột tối đa DQN/Pricing chọn 1 lần (để tránh bùng nổ cột)

'''
# --- MODEL PATHS ---
'''
# Model POMO dùng chung (Pricing Solver)
POMO_MODEL_PATH = os.path.join(DIR_POMO_OUTPUT, 'checkpoint-50.pt') # Checkpoint bạn đã train
# DQN Model Load path (Cho Testing Mode)
DQN_MODEL_LOAD_PATH = None # Sẽ override bằng arg parser

'''
# --- POMO PARAMETERS ---
'''
POMO_EMBEDDING_DIM = 128
POMO_ENCODER_LAYER_NUM = 6
POMO_QKV_DIM = 16
POMO_HEAD_NUM = 8
POMO_LOGIT_CLIPPING = 10
POMO_FF_HIDDEN_DIM = 512
POMO_EVAL_TYPE = 'argmax'

GNN_COL_FEAT_DIM = 8
GNN_CON_FEAT_DIM = 2
GNN_HIDDEN_DIM = 64
'''
# --- EXACT SOLVER PARAMETERS ---
'''
EXACT_MAX_ITE = 10000
'''
# --- TRAINING CONFIG ---
'''
TRAIN_INSTANCE_FILE = os.path.join(DATA_DIR, 'Cebu_depot_customer_with_schedules.csv')

DQN_TRAIN_EPISODES = 2      # Tổng số episode (bài toán) để train
DQN_MAX_STEPS_PER_EPISODE = 3 # Max CG loops per episode

DQN_LR = 1e-4
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY = 1000 # steps
MEMORY_CAPACITY = 10000 
BATCH_SIZE = 32       
TARGET_UPDATE_FREQ = 20

'''
# --- TESTING CONFIG ---
'''
# Test Instance: có thể là file khác hoặc file train
TEST_INSTANCE_FILE = os.path.join(DATA_DIR, 'Canlubang_depot_customer_with_schedules.csv')