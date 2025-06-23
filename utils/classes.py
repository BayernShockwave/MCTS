from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CRA:
    ecs_list: List[str]
    target_trains: List[str]
    target_locations: List[str]
    parameters: Dict[str, any]  # 具体动作, 例如STT多长时间或RRT哪个站台等
    conflict_index: int
    estimated_resolution_time: float

    def __hash__(self):
        return hash((
            tuple(self.ecs_list),
            tuple(self.target_trains),
            tuple(self.target_locations),
            tuple(sorted(self.parameters.items())),
            self.conflict_index,
            self.estimated_resolution_time
        ))

    def __eq__(self, other):
        if not isinstance(other, CRA):
            return False
        return (self.ecs_list == other.ecs_list and
                self.target_trains == other.target_trains and
                self.target_locations == other.target_locations and
                self.parameters == other.parameters and
                self.conflict_index == other.conflict_index and
                self.estimated_resolution_time == other.estimated_resolution_time)
