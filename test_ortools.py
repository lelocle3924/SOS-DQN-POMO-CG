# test_rmp_ortools.py

from envs.rmp_service import RMPService
from data.processor import process_instance_data
from configs import main_config as cfg

def run_test():
    print(f"Running OR-Tools RMP Test.")
    print(f"Data Path: {cfg.TEST_INSTANCE_FILE}")
    
    # 1. Load Data
    try:
        locations, d_mat, t_mat = process_instance_data(cfg.TEST_INSTANCE_FILE)
        n_cust = len(locations) - 1
        print(f"Data Loaded: {n_cust} customers.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Init RMP (Google OR-Tools)
    print("\n>>> Init RMP (GLOP backend)...")
    customers_df = locations[locations['Number'] != 0]
    
    rmp = RMPService(
        customers_df=customers_df, 
        vehicle_capacity=cfg.VEHICLE_CAPACITY, 
        big_m=cfg.RMP_BIG_M
    )
    
    # 3. Create Dummy Routes (Mỗi khách 1 xe)
    print(">>> Generating Initial Columns...")
    init_routes = []
    init_costs = []
    
    for _, row in customers_df.iterrows():
        cid = int(row['Number'])
        r = [0, cid, 0]
        
        # Lấy index ma trận
        idx_cid = locations.index.get_loc(cid)
        idx_depot = locations.index.get_loc(0)
        
        # Cost = Time đi + Service + Time về (dùng ma trận t_mat từ processor)
        cost = t_mat[idx_depot, idx_cid] + t_mat[idx_cid, idx_depot] + row['ServiceTime']
        
        init_routes.append(r)
        init_costs.append(cost)
        
    print(f"   Created {len(init_routes)} routes.")
    rmp.add_initial_columns(init_routes, init_costs)
    print("First 5 routes:", init_routes[:5])
    print("First 5 costs:", init_costs[:5])
    
    # 4. Solve
    print(">>> Solving RMP...")
    obj, duals, basis = rmp.solve()
    
    if obj:
        print(f"✅ Solved Successfully!")
        print(f"   Objective Value: {obj:.4f}")
        print(f"   Columns in basis: {len(basis)}")
        print(f"   First 5 Duals: {list(duals.values())[:5]}")
    else:
        print("❌ Solver Failed.")

if __name__ == "__main__":
    run_test()