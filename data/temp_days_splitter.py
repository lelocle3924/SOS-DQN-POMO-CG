import pandas as pd
import os
import shutil


def split_orders_by_date(original_csv_path, output_dir):
    """
    Đọc file Order tổng -> Group đơn trùng -> Tách ra file theo ngày.
    """
    print(f">>> [1/4] Splitting Orders from {original_csv_path}...")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Clear old temp files
    os.makedirs(output_dir)
    
    df = pd.read_csv(original_csv_path)
    
    # Normalize Date Column (Adjust 'Delivery Date' to match your CSV header)
    date_col = 'Delivery Date' 
    if date_col not in df.columns:
        # Fallback logic if column name differs
        possible_cols = [c for c in df.columns if 'date' in c.lower()]
        if possible_cols: date_col = possible_cols[0]
    
    depot_col = "Depot"
    depot = df[depot_col][1]
    print(depot)
    
    unique_dates = df[date_col].unique()
    generated_files = []
    
    print(f"    Found {len(unique_dates)} unique dates.")
    
    for d in unique_dates:
        # Filter Day
        day_df = df[df[date_col] == d].copy()
        
        # Consolidate Logic: Group by Customer to sum Demand
        # Giữ lại thông tin tĩnh (Lat, Long, TimeWindow...) của dòng đầu tiên
        agg_rules = {
            'KGM': 'sum', 'CBM': 'sum',
            'CusLat': 'first', 'CusLong': 'first',
            'Beginning1': 'first', 'Ending1': 'first',
            'DwellTime': 'first', 'AllowedTrucks': 'first',
            'Depot': 'first', 'DepotLat': 'first', 'DepotLong': 'first'
        }
        # Add other columns to 'first' if needed to prevent loss
        for col in day_df.columns:
            if col not in agg_rules and col not in ['Customer', date_col]:
                agg_rules[col] = 'first'
                
        consolidated_df = day_df.groupby('Customer', as_index=False).agg(agg_rules)
        
        # Save Temp File
        safe_date = str(d).replace("/", "-").replace(" ", "_")
        fname = os.path.join(output_dir, f"{depot}_{safe_date}.csv")
        consolidated_df.to_csv(fname, index=False)
        generated_files.append((safe_date, fname))
        
    print(f"    Generated {len(generated_files)} temp files in {output_dir}")
    return generated_files

if __name__ == "__main__":
    full_path = "Split_TransportOrder_2524.csv"
    results_path = "temp_days_2524"
    split_orders_by_date(full_path, results_path) 