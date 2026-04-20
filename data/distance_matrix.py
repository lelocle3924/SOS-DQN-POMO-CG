import pandas as pd
import requests
import numpy as np
import os
import polyline
import folium
import time
import json
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DistanceMatrixCalculator:
    def __init__(self, order_csv_path: str, truck_csv_path: str = None):
        """
        Args:
            order_csv_path: Đường dẫn file đơn hàng (để lấy tọa độ Nodes)
            truck_csv_path: Đường dẫn file xe (để lấy tốc độ tính Time Matrix)
        """
        self.order_csv_path = order_csv_path
        self.truck_csv_path = truck_csv_path
        
        self.osrm_table_url = "http://router.project-osrm.org/table/v1/driving/"
        self.osrm_route_url = "http://router.project-osrm.org/route/v1/driving/"
        
        # Load nodes ngay khi init
        self.nodes = self._load_and_parse_nodes()
        self._cached_distance_array_meters = None 
        
        self.BATCH_SIZE = 60
        self.REQUEST_DELAY = 1.0 
        self.TIMEOUT = 30
        self.ROUTE_REQUEST_DELAY = 0.2

    def _load_and_parse_nodes(self):
        try:
            df = pd.read_csv(self.order_csv_path)
        except Exception as e:
            raise Exception(f"Input error reading orders: {e}")

        depot_info = df.iloc[0]
        depot_node = {
            'ID': 'Depot',
            'Lat': float(depot_info['DepotLat']),
            'Long': float(depot_info['DepotLong']),
            'Type': 'Depot'
        }

        unique_customers = df.groupby('Customer').agg({
            'CusLat': 'first',
            'CusLong': 'first'
        }).reset_index()

        customer_nodes = []
        for _, row in unique_customers.iterrows():
            customer_nodes.append({
                'ID': str(int(row['Customer'])),
                'Lat': float(row['CusLat']),
                'Long': float(row['CusLong']),
                'Type': 'Customer'
            })

        return [depot_node] + customer_nodes

    def _calculate_haversine_subset(self, src_nodes, dst_nodes):
        matrix = np.zeros((len(src_nodes), len(dst_nodes)))
        for i, src in enumerate(src_nodes):
            for j, dst in enumerate(dst_nodes):
                lat1, lon1 = radians(src['Lat']), radians(src['Long'])
                lat2, lon2 = radians(dst['Lat']), radians(dst['Long'])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                matrix[i][j] = c * 6371 * 1000 * 1.3 
        return matrix

    def _fetch_chunk_with_retry(self, src_nodes, dst_nodes, max_retries=3):
        all_chunk_nodes = src_nodes + dst_nodes
        coords_str = ";".join([f"{n['Long']},{n['Lat']}" for n in all_chunk_nodes])
        
        src_indices = ";".join([str(i) for i in range(len(src_nodes))])
        dst_indices = ";".join([str(i + len(src_nodes)) for i in range(len(dst_nodes))])
        
        request_url = f"{self.osrm_table_url}{coords_str}?sources={src_indices}&destinations={dst_indices}&annotations=distance"
        
        for attempt in range(max_retries):
            try:
                time.sleep(self.REQUEST_DELAY)
                response = requests.get(request_url, timeout=self.TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    return np.array(data['distances'])
                elif response.status_code == 429:
                    print(f"    [429] Server busy. Retrying in {2*(attempt+1)}s...")
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"    [API Error] Status {response.status_code}. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"    [Net Error] {e}. Attempt {attempt+1}/{max_retries}")
                time.sleep(2 * (attempt + 1))
        return None

    def _fetch_full_matrix_batched(self):
        # ... (Logic fetch OSRM giữ nguyên không đổi) ...
        n = len(self.nodes)
        full_matrix = np.zeros((n, n))
        
        print(f"  > Starting batch processing for {n} nodes...")
        print(f"  > Config: Batch Size={self.BATCH_SIZE}, Delay={self.REQUEST_DELAY}s")
        
        for i in range(0, n, self.BATCH_SIZE):
            src_chunk = self.nodes[i : i + self.BATCH_SIZE]
            for j in range(0, n, self.BATCH_SIZE):
                dst_chunk = self.nodes[j : j + self.BATCH_SIZE]
                print(f"    - Processing Chunk: Rows {i}-{i+len(src_chunk)} | Cols {j}-{j+len(dst_chunk)}...", end="\r")
                chunk_matrix = self._fetch_chunk_with_retry(src_chunk, dst_chunk)
                if chunk_matrix is not None:
                    full_matrix[i : i + len(src_chunk), j : j + len(dst_chunk)] = chunk_matrix
                else:
                    print(f"\n    [CRITICAL] Chunk fail. Filling with Haversine.")
                    fallback_matrix = self._calculate_haversine_subset(src_chunk, dst_chunk)
                    full_matrix[i : i + len(src_chunk), j : j + len(dst_chunk)] = fallback_matrix
        
        print("\n  > Matrix download complete.")
        full_matrix = np.nan_to_num(full_matrix, nan=1e9)
        return full_matrix

    def _format_waypoint_key(self, latitude: float, longitude: float) -> str:
        """Use fixed precision so geometry keys match downstream consumers."""
        return f"{latitude:.5f},{longitude:.5f}"

    def _build_geometry_key(self, from_node: dict, to_node: dict) -> str:
        start_waypoint = self._format_waypoint_key(from_node['Lat'], from_node['Long'])
        end_waypoint = self._format_waypoint_key(to_node['Lat'], to_node['Long'])
        return f"{start_waypoint};{end_waypoint}"

    def _fetch_route_geometry_encoded(self, from_node: dict, to_node: dict, max_retries: int = 3):
        """Fetch encoded route geometry for one directed arc from OSRM."""
        request_coordinates = (
            f"{from_node['Long']},{from_node['Lat']};"
            f"{to_node['Long']},{to_node['Lat']}"
        )
        request_url = (
            f"{self.osrm_route_url}{request_coordinates}"
            "?overview=full&geometries=polyline"
        )

        for attempt in range(max_retries):
            try:
                time.sleep(self.ROUTE_REQUEST_DELAY)
                response = requests.get(request_url, timeout=self.TIMEOUT)
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("code") == "Ok" and response_data.get("routes"):
                        geometry_value = response_data["routes"][0].get("geometry")
                        if isinstance(geometry_value, str) and geometry_value:
                            return geometry_value
                elif response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                else:
                    time.sleep(attempt + 1)
            except requests.exceptions.RequestException:
                time.sleep(2 * (attempt + 1))

        return None

    def _build_straight_line_encoded_geometry(self, from_node: dict, to_node: dict) -> str:
        """Fallback geometry when OSRM route call fails."""
        return polyline.encode([
            (from_node['Lat'], from_node['Long']),
            (to_node['Lat'], to_node['Long']),
        ])

    def create_geometry_json(self, output_path: str):
        """
        Build geometry json as:
        {
          "lat1,lon1;lat2,lon2": "<encoded_polyline>",
          ...
        }
        """
        if not self.nodes:
            raise ValueError("No nodes available to build geometry JSON.")

        geometry_by_arc = {}
        node_count = len(self.nodes)
        total_arcs = node_count * node_count
        processed_arc_count = 0

        print(f"  > Building geometry JSON for {total_arcs} directed arcs...")
        for source_node in self.nodes:
            for destination_node in self.nodes:
                geometry_key = self._build_geometry_key(source_node, destination_node)

                if source_node['ID'] == destination_node['ID']:
                    encoded_geometry = self._build_straight_line_encoded_geometry(source_node, destination_node)
                else:
                    encoded_geometry = self._fetch_route_geometry_encoded(source_node, destination_node)
                    if encoded_geometry is None:
                        encoded_geometry = self._build_straight_line_encoded_geometry(source_node, destination_node)

                geometry_by_arc[geometry_key] = encoded_geometry
                processed_arc_count += 1
                if processed_arc_count % 100 == 0 or processed_arc_count == total_arcs:
                    print(f"    - Geometry progress: {processed_arc_count}/{total_arcs}", end="\r")

        print()
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(geometry_by_arc, output_file)

        print(f"  > Saved Geometry JSON: {output_path}")

    def _load_fleet_speeds(self):
        """Helper để đọc file TruckMaster và lấy danh sách tốc độ"""
        if not self.truck_csv_path or not os.path.exists(self.truck_csv_path):
            print("  [Warning] Truck CSV path missing. Using default speed 30km/h for all.")
            return [40.0] # Default fallback

        try:
            df_truck = pd.read_csv(self.truck_csv_path)
            # Giả định index của row tương ứng với type_id (giống logic RealDataLoader)
            speeds = df_truck['AverageSpeedKmH'].astype(float).tolist()
            return speeds
        except Exception as e:
            print(f"  [Error] Reading truck speeds: {e}")
            return [40.0]

    def calculate_matrices(self):
        """
        Returns:
            dist_matrix_meters (np.ndarray): 2D array [N, N]
            super_time_matrix (np.ndarray): 3D array [NumVehicles, N, N] (Minutes)
        """
        # 1. Get Distance Matrix (Meters)
        if self._cached_distance_array_meters is None:
            self._cached_distance_array_meters = self._fetch_full_matrix_batched()
        else:
            print("  > Using cached raw distances...")

        dist_array = self._cached_distance_array_meters
        
        # 2. Get Fleet Speeds
        speeds_kmh = self._load_fleet_speeds()
        num_vehicle_types = len(speeds_kmh)
        num_nodes = dist_array.shape[0]
        
        # 3. Build Super Time Matrix
        print(f"  > Building Super Time Matrix for {num_vehicle_types} vehicle types...")
        super_time_matrix = np.zeros((num_vehicle_types, num_nodes, num_nodes))

        for idx, speed_kmh in enumerate(speeds_kmh):
            # Convert km/h -> m/min
            speed_mpm = speed_kmh * 1000.0 / 60.0
            if speed_mpm <= 0.1:
                speed_mpm = 0.1 # Safety
            
            # Time (min) = Distance (m) / Speed (m/min)
            time_matrix_v = dist_array / speed_mpm
            super_time_matrix[idx] = time_matrix_v
            
        return dist_array, super_time_matrix

    def export_demo_map(self, output_file='real_route.html'):
        if not self.nodes: return
        depot = self.nodes[0]
        m = folium.Map(location=[depot['Lat'], depot['Long']], zoom_start=12, tiles='CartoDB positron')
        bounds = []
        for node in self.nodes:
            loc = [node['Lat'], node['Long']]
            bounds.append(loc)
            if node['Type'] == 'Depot':
                folium.Marker(loc, popup=f"Depot: {node['ID']}", icon=folium.Icon(color='red', icon='home')).add_to(m)
            else:
                folium.CircleMarker(loc, radius=3, color='blue', fill=True, fill_opacity=0.7).add_to(m)
        m.fit_bounds(bounds)
        m.save(output_file)
        print(f"  > Map exported: {output_file}")


if __name__ == "__main__":
    ORDER_FILE = "Split_TransportOrder_2510.csv"
    DEPOT = ORDER_FILE[-8:-4]
    TRUCK_FILE = "TruckMaster.csv"
    
    OUTPUT_DIR = "distmatrix_2510"
    DRAW_MAP = True
    EXPORT_GEOMETRY = True
    
    if os.path.exists(ORDER_FILE) and os.path.exists(TRUCK_FILE):
        print(">>> DISTANCE MATRIX GENERATOR (STANDALONE TEST) <<<")
        
        # 1. Khởi tạo với 2 tham số
        calculator = DistanceMatrixCalculator(ORDER_FILE, TRUCK_FILE)
        
        # 2. Tính toán (Lưu ý: super_time_matrix là mảng 3 chiều [VehType, Node, Node])
        dist_matrix, super_time_matrix = calculator.calculate_matrices()
        
        # 3. Lấy Node IDs để làm nhãn cho DataFrame
        node_ids = [n['ID'] for n in calculator.nodes]
        
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
        # 4. Xuất Distance Matrix
        df_dist = pd.DataFrame(dist_matrix, index=node_ids, columns=node_ids)
        df_dist.to_csv(f'{OUTPUT_DIR}/distance_matrix_meters{DEPOT}.csv')
        print(f"  > Saved Distance Matrix: {df_dist.shape}")
        
        # 5. Xuất Time Matrix (Vì là 3D nên ta xuất từng layer ra từng file CSV riêng)
        num_vehicle_types = super_time_matrix.shape[0]
        print(f"  > Saving Time Matrices for {num_vehicle_types} vehicle types...")
        
        for i in range(num_vehicle_types):
            time_layer = super_time_matrix[i]
            df_time = pd.DataFrame(np.squeeze(time_layer), index=node_ids, columns=node_ids)
            # Lưu file theo index loại xe (Type 0 = xe đầu tiên trong TruckMaster)
            df_time.to_csv(f'{OUTPUT_DIR}/time_matrix_type_{i}_mins{DEPOT}.csv')

        if EXPORT_GEOMETRY:
            geometry_output_path = f'{OUTPUT_DIR}/geometry_{DEPOT}.json'
            calculator.create_geometry_json(geometry_output_path)
            
        # 6. Vẽ Map Demo
        if DRAW_MAP:
            calculator.export_demo_map(f'{OUTPUT_DIR}/map_visualization{DEPOT}.html')
            
        print(">>> Processing complete. <<<")
    else:
        print(f"❌ Critical files missing.")
        print(f"   Order File exists: {os.path.exists(ORDER_FILE)}")
        print(f"   Truck File exists: {os.path.exists(TRUCK_FILE)}")