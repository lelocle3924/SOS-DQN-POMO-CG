"""Load and preprocess orders, trucks, and distance matrix from CSV files."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.utils import parse_time_string, parse_allowed_trucks


class ProblemData:
    """Container for all preprocessed SD-VRPTW problem data.

    Node indexing convention used throughout the codebase:
        node 0        = depot
        node c+1      = customer with 0-based customer index c
    """

    def __init__(self) -> None:
        self.num_customers: int = 0
        self.depot_id: str = ""
        self.customer_ids: List[str] = []
        self.node_ids: List[str] = []

        # Per-customer arrays  (length = num_customers, 0-indexed)
        self.demands: np.ndarray = np.array([])
        self.coords: np.ndarray = np.array([])       # (num_customers, 2)
        self.tw_start: np.ndarray = np.array([])      # hours
        self.tw_end: np.ndarray = np.array([])         # hours
        self.service_times: np.ndarray = np.array([])  # hours
        self.allowed_trucks: List[set] = []

        # Depot
        self.depot_coord: np.ndarray = np.array([])
        self.depot_tw_start: float = 0.0
        self.depot_tw_end: float = 24.0

        # All nodes  (depot first, then customers)
        self.all_coords: np.ndarray = np.array([])  # (num_nodes, 2)

        # Vehicle fleet
        self.vehicle_types: List[str] = []
        self.vehicle_capacity: Dict[str, float] = {}
        self.vehicle_fixed_cost: Dict[str, float] = {}
        self.vehicle_cost_per_km: Dict[str, float] = {}
        self.vehicle_cost_per_hour: Dict[str, float] = {}
        self.vehicle_speed_kmh: Dict[str, float] = {}
        self.vehicle_count: Dict[str, int] = {}

        # Distance / time matrices   (num_nodes x num_nodes, node 0 = depot)
        self.distance_matrix_meters: np.ndarray = np.array([])
        self.travel_time_matrices: Dict[str, np.ndarray] = {}

        # Site dependency  (num_customers x num_vehicle_types) boolean
        self.site_dependency: np.ndarray = np.array([])


@dataclass
class TrainingInstanceSpec:
    """Metadata describing one depot/day training instance."""

    orders_file: str
    distance_matrix_file: str
    depot_id: str
    customer_count: int
    instance_name: str


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def load_problem(orders_path: str, trucks_path: str,
                 distance_matrix_path: str,
                 depot_id: str = "2524") -> ProblemData:
    """Load all CSVs and build a fully-populated ProblemData instance."""
    data = ProblemData()
    data.depot_id = depot_id

    _load_orders(data, orders_path)
    _load_trucks(data, trucks_path)
    _load_distance_matrix(data, distance_matrix_path)
    _build_travel_time_matrices(data)
    _build_site_dependency_matrix(data)

    return data


def build_training_manifest(
    temp_day_dirs: List[str],
    distance_matrix_dirs: List[str],
    distance_matrix_pattern: str = "distance_matrix_meters{depot_id}.csv",
) -> List[TrainingInstanceSpec]:
    """Build a depot-aware manifest for mixed-map training.

    Parameters
    ----------
    temp_day_dirs:
        List of directories containing depot/day CSV files.
    distance_matrix_dirs:
        List of directories containing depot distance matrices.
    distance_matrix_pattern:
        Filename pattern where ``{depot_id}`` is substituted.
    """
    if len(temp_day_dirs) != len(distance_matrix_dirs):
        raise ValueError("temp_day_dirs and distance_matrix_dirs must have equal length.")

    manifest: List[TrainingInstanceSpec] = []
    for temp_dir, matrix_dir in zip(temp_day_dirs, distance_matrix_dirs):
        temp_path = Path(temp_dir)
        matrix_path = Path(matrix_dir)
        if not temp_path.is_dir():
            raise ValueError(f"Temp-day directory does not exist: {temp_path}")
        if not matrix_path.is_dir():
            raise ValueError(f"Distance-matrix directory does not exist: {matrix_path}")

        for orders_file in sorted(temp_path.glob("*.csv")):
            depot_id = _extract_depot_id_from_filename(orders_file.name)
            if depot_id is None:
                continue
            matrix_name = distance_matrix_pattern.format(depot_id=depot_id)
            matrix_file = matrix_path / matrix_name
            if not matrix_file.is_file():
                # Fallback: first matrix file in folder if exact pattern is unavailable.
                fallback = sorted(matrix_path.glob("distance_matrix_meters*.csv"))
                if not fallback:
                    raise ValueError(f"No distance matrix found in: {matrix_path}")
                matrix_file = fallback[0]

            customer_count = _safe_count_customers(str(orders_file))
            manifest.append(
                TrainingInstanceSpec(
                    orders_file=str(orders_file),
                    distance_matrix_file=str(matrix_file),
                    depot_id=depot_id,
                    customer_count=customer_count,
                    instance_name=orders_file.name,
                )
            )

    if not manifest:
        raise ValueError("No training instances found for manifest.")
    return manifest


def load_problem_for_training_instance(
    instance_spec: TrainingInstanceSpec,
    trucks_path: str,
) -> ProblemData:
    """Load one manifest instance into a ProblemData object."""
    problem = load_problem(
        orders_path=instance_spec.orders_file,
        trucks_path=trucks_path,
        distance_matrix_path=instance_spec.distance_matrix_file,
        depot_id=instance_spec.depot_id,
    )
    _validate_problem_matrix_coverage(problem, instance_spec)
    return problem


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_orders(data: ProblemData, path: str) -> None:
    df = pd.read_csv(path)

    data.num_customers = len(df)
    data.customer_ids = [str(c) for c in df["Customer"].tolist()]
    data.node_ids = [data.depot_id] + data.customer_ids

    data.demands = df["KGM"].values.astype(np.float64)
    data.coords = df[["CusLat", "CusLong"]].values.astype(np.float64)
    data.tw_start = np.array([parse_time_string(t) for t in df["Beginning1"]])
    data.tw_end = np.array([parse_time_string(t) for t in df["Ending1"]])
    data.service_times = df["DwellTime"].values.astype(np.float64)
    data.allowed_trucks = [parse_allowed_trucks(s) for s in df["AllowedTrucks"]]

    data.depot_coord = np.array(
        [df["DepotLat"].iloc[0], df["DepotLong"].iloc[0]], dtype=np.float64
    )
    data.depot_tw_start = 0.0
    data.depot_tw_end = 24.0

    data.all_coords = np.vstack([data.depot_coord.reshape(1, 2), data.coords])


def _load_trucks(data: ProblemData, path: str) -> None:
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        name = row["TruckName"]
        data.vehicle_types.append(name)
        data.vehicle_capacity[name] = float(row["CapacityKg"])
        data.vehicle_fixed_cost[name] = float(row["FixedCost"])
        data.vehicle_cost_per_km[name] = float(row["CostPerKm"])
        data.vehicle_cost_per_hour[name] = float(row["CostPerHour"])
        data.vehicle_speed_kmh[name] = float(row["AverageSpeedKmH"])
        # Default to a large number if Count is missing or unconstrained
        if "Count" in row and pd.notna(row["Count"]):
            data.vehicle_count[name] = int(row["Count"])
        else:
            data.vehicle_count[name] = 999999


def _load_distance_matrix(data: ProblemData, path: str) -> None:
    """Extract the submatrix for (depot + order customers) from the full matrix."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    num_nodes = len(data.node_ids)
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    def _resolve_matrix_node_id(node_id: str, is_depot: bool, available_values) -> str:
        if node_id in available_values:
            return node_id
        if is_depot:
            for alias in ("Depot", "depot", "DEPOT"):
                if alias in available_values:
                    return alias
        return node_id

    for i, node_i in enumerate(data.node_ids):
        for j, node_j in enumerate(data.node_ids):
            matrix_i = _resolve_matrix_node_id(node_i, is_depot=(i == 0), available_values=df.index)
            matrix_j = _resolve_matrix_node_id(node_j, is_depot=(j == 0), available_values=df.columns)
            if matrix_i in df.index and matrix_j in df.columns:
                matrix[i, j] = df.loc[matrix_i, matrix_j]
            elif i != j:
                raise ValueError(
                    f"Missing distance for ({node_i}, {node_j}) in the matrix."
                )

    data.distance_matrix_meters = matrix


def _build_travel_time_matrices(data: ProblemData) -> None:
    """travel_time_hours = distance_km / speed_kmh  for every vehicle type."""
    distance_km = data.distance_matrix_meters / 1000.0
    for vtype in data.vehicle_types:
        speed = data.vehicle_speed_kmh[vtype]
        data.travel_time_matrices[vtype] = distance_km / speed


def _build_site_dependency_matrix(data: ProblemData) -> None:
    """site_dependency[c, v] = True iff vehicle v is allowed to visit customer c."""
    num_c = data.num_customers
    num_v = len(data.vehicle_types)
    data.site_dependency = np.zeros((num_c, num_v), dtype=bool)

    for c_idx in range(num_c):
        allowed = data.allowed_trucks[c_idx]
        if not allowed:
            # Missing allowed-truck data: fall back to all vehicle types.
            allowed = set(data.vehicle_types)
        for v_idx, vtype in enumerate(data.vehicle_types):
            data.site_dependency[c_idx, v_idx] = vtype in allowed


def _extract_depot_id_from_filename(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if not parts:
        return None
    if parts[0].isdigit() and len(parts[0]) == 4:
        return parts[0]
    return None


def _safe_count_customers(orders_path: str) -> int:
    df = pd.read_csv(orders_path)
    return int(len(df))


def _validate_problem_matrix_coverage(
    problem: ProblemData,
    instance_spec: TrainingInstanceSpec,
) -> None:
    expected_shape = (problem.num_customers + 1, problem.num_customers + 1)
    if problem.distance_matrix_meters.shape != expected_shape:
        raise ValueError(
            f"Distance matrix mismatch for {instance_spec.instance_name}: "
            f"expected {expected_shape}, got {problem.distance_matrix_meters.shape}"
        )
