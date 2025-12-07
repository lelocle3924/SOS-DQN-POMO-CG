import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from configs import main_config as cfg

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 # Earth radius in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c 

def parse_time_to_minutes(time_str):
    if pd.isna(time_str): return 0.0
    try:
        parts = str(time_str).strip().split(':')
        h, m = int(parts[0]), int(parts[1])
        return h * 60 + m
    except:
        return 0.0

def process_instance_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instance file not found: {file_path}")

    df = pd.read_csv(file_path)
    
    # 1. Clean Data & Sort
    if 'Number' not in df.columns:
        # Fallback: Nếu không có Number nhưng có ID
        if 'ID' in df.columns:
            print("[Warning] 'Number' column missing, creating one from reset_index.")
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Number'}, inplace=True)
        else:
            raise ValueError("CSV missing 'Number' or 'ID' column")
        
    df.sort_values(by='Number', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Depot Check
    if df.iloc[0]['Number'] != 0:
        pass 

    # 2. Time Window Parsing
    time_cols = ['Beginning', 'Ending']
    if all(col in df.columns for col in time_cols):
        df['Start_Time'] = df['Beginning'].apply(parse_time_to_minutes)
        df['End_Time'] = df['Ending'].apply(parse_time_to_minutes)
    else:
        df['Start_Time'] = 0.0
        df['End_Time'] = 24 * 60.0
    
    # 3. Default Values form Config
    if 'ServiceTime' not in df.columns:
        df['ServiceTime'] = cfg.DEFAULT_SERVICE_TIME
        
    if 'Demand' not in df.columns:
        df['Demand'] = cfg.DEFAULT_DEMAND
        
    # Tạo bản sao
    locations = df.copy().set_index('Number', drop=False)

    # 4. Generate Matrices
    coords = locations[['Latitude', 'Longitude']].values
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine_distance(
                    coords[i, 0], coords[i, 1], 
                    coords[j, 0], coords[j, 1]
                )
    
    # Time Matrix Generation from Config Speed
    # speed (km/h) -> speed (km/min)
    speed_km_min = cfg.VEHICLE_SPEED_KMH / 60.0
    time_matrix = dist_matrix / speed_km_min
    
    return locations, dist_matrix, time_matrix