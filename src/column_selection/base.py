"""Common interfaces for CG column-selection policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.column_pool import ColumnPool, Route
from src.data_loader import ProblemData
from src.master_problem import MasterProblemResult


@dataclass
class ColumnSelectionState:
    """State passed from CG loop to column-selection policies."""

    problem: ProblemData
    column_pool: ColumnPool
    rmp_result: MasterProblemResult
    iteration_index: int
    forbidden_arcs: Set[Tuple[int, int]]
    enforced_arcs: Set[Tuple[int, int]]
    config: Dict


class AbstractColumnSelector(ABC):
    """Abstract selector used by CG loop."""

    @abstractmethod
    def select_columns(
        self,
        state: ColumnSelectionState,
        candidate_routes: Sequence[Route],
    ) -> List[Route]:
        """Return final route(s) to add to the RMP."""

    def update_after_iteration(
        self,
        previous_state: ColumnSelectionState,
        selected_routes: Sequence[Route],
        new_rmp_result: Optional[MasterProblemResult],
    ) -> None:
        """Optional hook for online-learning selectors."""
        return
