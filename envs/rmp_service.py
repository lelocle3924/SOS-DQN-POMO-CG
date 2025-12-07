from ortools.linear_solver import pywraplp
import numpy as np

class RMPService:
    """
    Persistent Restricted Master Problem Solver sử dụng Google OR-Tools (GLOP Backend).
    Solver này hoàn toàn miễn phí, open-source và hỗ trợ cơ chế Warm-Start tốt cho LP.
    """
    def __init__(self, customers_df, vehicle_capacity, big_m=1e5):
        self.customers_df = customers_df
        # Lấy Customer ID (bỏ Depot)
        self.customer_ids = [idx for idx in self.customers_df.index if idx != 0]
        self.vehicle_capacity = vehicle_capacity
        
        # 1. Khởi tạo Solver (GLOP là Solver LP chuẩn của Google)
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        if not self.solver:
            raise Exception("Google OR-Tools GLOP backend not available.")
            
        # Infinity constant
        self.infinity = self.solver.infinity()
        
        # 2. Storage
        self.constrs = {}      # Map: CID -> OR-Tools Constraint Object
        self.vars = []         # List chứa OR-Tools Variable Objects (Routes)
        self.routes_data = []  
        self.column_costs = [] 
        
        # 3. Khởi tạo RMP (Constraints + Artificial Vars)
        self._initialize_rmp(big_m)

    def _initialize_rmp(self, big_m):
        """Khởi tạo Constraints Set Partitioning với Artificial Variables (Big-M)."""
        # Objective (Minimize) -> Sẽ cộng dồn các biến vào đây
        # Trong OR-Tools, ta set coefficient cho objective function trên từng biến
        
        objective = self.solver.Objective()
        objective.SetMinimization()
        
        for cid in self.customer_ids:
            # Tạo biến giả (Artificial Var): Cost = Big_M, Bound >= 0
            art_var_name = f"Artific_{cid}"
            art_var = self.solver.NumVar(0.0, self.infinity, art_var_name)
            
            # Thêm vào Objective: hệ số là Big_M
            objective.SetCoefficient(art_var, big_m)
            
            # Tạo Constraint: sum(lambda) + artific == 1
            # Bound: 1 <= expr <= 1 (Equality)
            ct_name = f"Serve_{cid}"
            ct = self.solver.Constraint(1.0, 1.0, ct_name)
            
            # Thêm Artificial Var vào Constraint này (hệ số 1.0)
            ct.SetCoefficient(art_var, 1.0)
            
            # Lưu tham chiếu Constraint để dùng sau này
            self.constrs[cid] = ct

    def add_initial_columns(self, routes, costs):
        for route, cost in zip(routes, costs):
            self.add_column(route, cost)

    def add_column(self, route, cost):
        """
        Thêm một cột (Route) mới vào Solver.
        Logic: Tạo biến mới -> Update hệ số của biến đó trong các Constraint liên quan.
        """
        nodes_in_route = [node for node in route if node != 0]
        
        valid_route = False
        
        # Kiểm tra sơ bộ xem route có đóng góp gì vào constraint hiện tại không
        target_constrs = []
        for cid in nodes_in_route:
            if cid in self.constrs:
                target_constrs.append(self.constrs[cid])
                valid_route = True
        
        if not valid_route:
            return

        # 1. Tạo biến Route mới (Continuous, >= 0)
        # GLOP giải LP nên mặc định biến là Continuous
        var_idx = len(self.vars)
        new_var = self.solver.NumVar(0.0, self.infinity, f"Route_{var_idx}")
        
        # 2. Set Objective Coefficient (Reduced Cost gốc = cost của route)
        self.solver.Objective().SetCoefficient(new_var, cost)
        
        # 3. Add to Constraints (Set Partitioning: hệ số a_ir = 1.0)
        # "SetCoefficient" trong OR-Tools tương đương việc điền số vào cột ma trận
        for ct in target_constrs:
            ct.SetCoefficient(new_var, 1.0)
            
        # 4. Save metadata
        self.vars.append(new_var)
        self.routes_data.append(route)
        self.column_costs.append(cost)

    def solve(self):
        """Solve và lấy Duals."""
        # Gọi Solver
        status = self.solver.Solve()
        
        # Check Status
        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            # GLOP rất hiếm khi fail nếu đã có biến giả
            print(f"!! RMP Solver Abnormal Status: {status}")
            return None, None, None
            
        obj_val = self.solver.Objective().Value()
        
        # 1. Get Duals (Shadow Price)
        # OR-Tools constraint object có hàm .dual_value()
        duals = {cid: ct.dual_value() for cid, ct in self.constrs.items()}
        
        # 2. Get Primal Values (Solution Basis)
        basis_values = {}
        for idx, var in enumerate(self.vars):
            val = var.solution_value()
            if val > 1e-6:
                basis_values[idx] = val
                
        return obj_val, duals, basis_values

    def get_route(self, index):
        if 0 <= index < len(self.routes_data):
            return self.routes_data[index]
        return None