import logging
import argparse
import os
import sys
from configs import main_config as cfg
import matplotlib.pyplot as plt
import folium
import numpy as np
import torch
from datetime import datetime

################################################
# Logging
################################################
def create_logger(log_file_path):
    """
    Tạo Logger object ghi đồng thời ra màn hình (Console) và File.
    Nếu folder chứa log file chưa có, tự tạo luôn.
    """
    # 1. Ensure dir exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('VRPCG_Logger')
    
    # Xóa handlers cũ nếu có (để tránh double log khi chạy notebook nhiều lần)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    
    # 2. Format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 3. File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 4. Stream Handler (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="RL Column Generation Solver")
    
    parser.add_argument("-m", "--mode", type=str, choices=["train", "test"], default="test",
                        help="Chạy Train (Học agent) hay Test (Giải bài toán cụ thể)")
    
    parser.add_argument("--episodes", type=int, default=cfg.DQN_TRAIN_EPISODES,
                        help="Số lượng bài toán (episodes) dùng để train")
    
    parser.add_argument("--test_file", type=str, default=cfg.TEST_INSTANCE_FILE,
                        help="Đường dẫn file instance CSV để test")
    
    parser.add_argument("--model_path", type=str, default=cfg.DQN_MODEL_LOAD_PATH,
                        help="Đường dẫn model DQN (.pt) để load khi test")
    
    return parser.parse_args()

################################################
# Checkpointing
################################################

def get_checkpoint_path(folder, prefix, steps, ext=".pt"):
    """
    Tạo đường dẫn file: output/dqn_model/checkpoint_1202_5000.pt
    """
    # Lấy ngày tháng hiện tại (DDMM)
    date_str = datetime.now().strftime("%d%m")
    
    filename = f"{prefix}_{date_str}_{steps}{ext}"
    full_path = os.path.join(folder, filename)
    
    # Đảm bảo folder tồn tại
    os.makedirs(folder, exist_ok=True)
    
    return full_path

def save_checkpoint(model, optimizer, steps, folder, prefix="checkpoint"):
    """
    Lưu model pytorch chuẩn
    """
    path = get_checkpoint_path(folder, prefix, steps)
    
    state = {
        'steps': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    
    torch.save(state, path)
    print(f"[Checkpoint] Saved to {path}")
    return path

################################################
# Visualization
################################################
import matplotlib.pyplot as plt
import os
import folium
import numpy as np

def visualize_solution(routes, locations_df, save_dir, title_suffix=""):
    """
    routes: List of lists [ [0, 5, 2, 0], [0, ...], ... ] (Node IDs)
    locations_df: DataFrame with Lat/Lon/Number
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Mapping ID -> Coords
    loc_map = {}
    depot_loc = None
    for _, row in locations_df.iterrows():
        nid = int(row['Number'])
        coords = (row['Longitude'], row['Latitude']) # X=Lon, Y=Lat
        loc_map[nid] = coords
        if nid == 0:
            depot_loc = (row['Latitude'], row['Longitude']) # For Folium (Lat, Lon)

    # 1. Matplotlib Plot
    plt.figure(figsize=(10, 8))
    
    # Draw Customers
    cust_x = [v[0] for k, v in loc_map.items() if k != 0]
    cust_y = [v[1] for k, v in loc_map.items() if k != 0]
    plt.scatter(cust_x, cust_y, c='blue', s=20, label='Customer')
    
    # Draw Depot
    depot = loc_map[0]
    plt.scatter([depot[0]], [depot[1]], c='red', s=100, marker='*', label='Depot')
    
    # Colors
    colors = plt.cm.get_cmap('tab10', len(routes))
    
    for idx, route in enumerate(routes):
        path_x = []
        path_y = []
        for node_id in route:
            pos = loc_map.get(node_id)
            if pos:
                path_x.append(pos[0])
                path_y.append(pos[1])
        
        plt.plot(path_x, path_y, '-', color=colors(idx % 10), alpha=0.7)
        
    plt.title(f"VRP Solution Routes {title_suffix}")
    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    plt.savefig(os.path.join(save_dir, f"solution_map{title_suffix}.png"))
    plt.close()

    # 2. Folium HTML Map
    m = folium.Map(location=depot_loc, zoom_start=13)
    
    # Depot Marker
    folium.Marker(depot_loc, popup='Depot', icon=folium.Icon(color='red')).add_to(m)
    
    # Colors hex
    colors_hex = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                  'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                  'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                  'gray', 'black', 'lightgray']
    
    for idx, route in enumerate(routes):
        route_coords = []
        # Convert Lon,Lat -> Lat,Lon for Folium
        for nid in route:
            lx, ly = loc_map[nid] # Lon, Lat
            route_coords.append((ly, lx)) 
            
            if nid != 0:
                folium.CircleMarker(
                    location=(ly, lx),
                    radius=3,
                    color=colors_hex[idx % len(colors_hex)],
                    popup=f"Cust {nid}",
                    fill=True
                ).add_to(m)
        
        folium.PolyLine(
            route_coords,
            color=colors_hex[idx % len(colors_hex)],
            weight=2.5,
            opacity=0.8,
            tooltip=f"Route {idx}"
        ).add_to(m)
        
    m.save(os.path.join(save_dir, f"solution_map{title_suffix}.html"))