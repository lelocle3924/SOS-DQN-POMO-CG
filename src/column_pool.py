"""Column (route) pool management for the master problem."""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np


@dataclass
class Route:
    """A single vehicle route: depot -> customers -> depot."""
    route_id: int
    vehicle_type: str
    customer_indices: List[int]   # 0-based customer indices
    visit_sequence: List[int]     # ordered node indices visited (excluding depot)
    total_cost: float
    total_distance_km: float
    total_time_hours: float
    reduced_cost: Optional[float] = None


class ColumnPool:
    """De-duplicated pool of route columns."""

    def __init__(self) -> None:
        self._routes: List[Route] = []
        self._next_id: int = 0
        self._seen: Set[Tuple] = set()

    @property
    def routes(self) -> List[Route]:
        return list(self._routes)

    @property
    def num_routes(self) -> int:
        return len(self._routes)

    def add_route(
        self,
        vehicle_type: str,
        customer_indices: List[int],
        total_cost: float,
        total_distance_km: float,
        total_time_hours: float,
        visit_sequence: Optional[List[int]] = None,
        reduced_cost: Optional[float] = None,
    ) -> Optional[Route]:
        """Add a route if not already present.  Returns the Route or None."""
        fingerprint = (vehicle_type, tuple(sorted(customer_indices)))
        if fingerprint in self._seen:
            return None

        route = Route(
            route_id=self._next_id,
            vehicle_type=vehicle_type,
            customer_indices=list(customer_indices),
            visit_sequence=list(visit_sequence or customer_indices),
            total_cost=total_cost,
            total_distance_km=total_distance_km,
            total_time_hours=total_time_hours,
            reduced_cost=reduced_cost,
        )
        self._routes.append(route)
        self._seen.add(fingerprint)
        self._next_id += 1
        return route

    # ------------------------------------------------------------------
    # Matrices consumed by the master problem
    # ------------------------------------------------------------------

    def get_coverage_matrix(self, num_customers: int) -> np.ndarray:
        """Binary A matrix: A[i, k] = 1 if route k covers customer i."""
        num_routes = len(self._routes)
        coverage = np.zeros((num_customers, num_routes), dtype=np.float64)
        for k, route in enumerate(self._routes):
            for c_idx in route.customer_indices:
                coverage[c_idx, k] = 1.0
        return coverage

    def get_cost_vector(self) -> np.ndarray:
        """Cost vector c[k] for each route."""
        return np.array([r.total_cost for r in self._routes], dtype=np.float64)
