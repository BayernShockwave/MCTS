from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ECS:
    name: str
    applicable_to: List[str]  # ['NODE', 'LINK'] or ['NODE]
    penalty: float


ECS_DEFINITIONS = {
    'STT': ECS('STT', ['NODE', 'LINK'], 10),
    'RRT': ECS('RRT', ['NODE'], 20),
    'TCT': ECS('TCT', ['NODE'], 160),
    'PCT': ECS('PCT', ['NODE'], 80),
    'ETT': ECS('ETT', ['NODE'], 40),
}


@dataclass
class CRA:
    ecs_list: List[str]
    target_trains: List[str]
    target_locations: List[str]
    parameters: Dict[str, any]  # 具体动作, 例如STT多长时间或RRT哪个站台等
    conflict_index: int
    estimated_resolution_time: float
