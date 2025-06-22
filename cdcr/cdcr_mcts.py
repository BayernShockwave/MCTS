import pandas as pd
from typing import List, Dict, Optional
from utils.classes import CRA
from data.load_data import NODE, HEADWAY
from utils.time_operation import seconds_to_hhmmss
import numpy as np


class ConflictState:
    def __init__(self, realized_schedule: pd.DataFrame, conflicts: List[Dict], applied_cras: List[CRA] = None, depth: int = 0):
        self.realized_schedule = realized_schedule.copy()
        self.conflicts = conflicts
        self.applied_cras = applied_cras if applied_cras is not None else []
        self.depth = depth
        self.terminal = len(conflicts) == 0 or depth >= 10

    def get_legal_actions(self) -> List[CRA]:
        if self.terminal or not self.conflicts:
            return []
        actions = []
        for idx, conflict in enumerate(self.conflicts):
            # STT
            for train_id in [conflict['course_id_1'], conflict['course_id_2']]:
                for delay in range(60, 600, 60):
                    cra = CRA(
                        ecs_list=['STT'],
                        target_trains=[train_id],
                        target_locations=[conflict['start_node']],
                        parameters={'delay_seconds': delay},
                        conflict_index=idx,
                        estimated_resolution_time=delay
                    )
                    actions.append(cra)
            # RRT
            if conflict['place'] == 'NODE' and conflict['detail'] != 'N/A':
                node_info = NODE[NODE['NODE'] == conflict['start_node']]
                available_tracks = []
                for _, row in node_info.iterrows():
                    if row['Track_EB'] and row['Track_EB'] != conflict['detail']:
                        available_tracks.append(row['Track_EB'])
                    if row['Track_WB'] and row['Track_WB'] != conflict['detail']:
                        available_tracks.append(row['Track_WB'])
                for train_id in [conflict['course_id_1'], conflict['course_id_2']]:
                    for track in available_tracks:
                        cra = CRA(
                            ecs_list=['RRT'],
                            target_trains=[train_id],
                            target_locations=[conflict['start_node']],
                            parameters={'new_track': track},
                            conflict_index=idx,
                            estimated_resolution_time=180  # TODO: 每与原轨道相差一个轨道, 则增加3分钟, 例如从轨道1改到轨道3, 需要6分钟
                        )
                        actions.append(cra)
        return actions

    def apply_action(self, cra: CRA) -> 'ConflictState':
        new_schedule = self.realized_schedule.copy()
        for ecs in cra.ecs_list:
            if ecs == 'STT':
                delay = cra.parameters['delay_seconds']
                for train_id in cra.target_trains:
                    # 从目标位置开始的所有后续站点都要延迟
                    mask = new_schedule['TRAIN_COURSE_ID'] == train_id
                    target_seq = new_schedule[mask & (new_schedule['NODE'].isin(cra.target_locations))]['SEQ'].min()
                    delay_mask = mask & (new_schedule['SEQ'] >= target_seq)
                    new_schedule.loc[delay_mask, 'ARRIVAL_SECONDS'] = new_schedule.loc[delay_mask, 'ARRIVAL_SECONDS'].apply(lambda x: x + delay if pd.notna(x) else x)
                    new_schedule.loc[delay_mask, 'DEPARTURE_SECONDS'] += delay
                    for idx in new_schedule[delay_mask].index:
                        if pd.notna(new_schedule.loc[idx, 'ARRIVAL_SECONDS']):
                            new_schedule.loc[idx, 'ARRIVAL_HHMMSS'] = seconds_to_hhmmss(new_schedule.loc[idx, 'ARRIVAL_SECONDS'])
                            new_schedule.loc[idx, 'DEPARTURE_HHMMSS'] = seconds_to_hhmmss(new_schedule.loc[idx, 'DEPARTURE_SECONDS'])
            elif ecs == 'RRT':
                new_track = cra.parameters['new_track']
                for train_id in cra.target_trains:
                    for location in cra.target_locations:
                        mask = (new_schedule['TRAIN_COURSE_ID'] == train_id) & (new_schedule['NODE'] == location)
                        new_schedule.loc[mask, 'TRACK'] = new_track
        detector = ConflictDetector(new_schedule)
        new_conflicts = detector.detect_all_conflicts()
        new_applied_cras = self.applied_cras + [cra]
        new_state = ConflictState(
            new_schedule,
            new_conflicts,
            new_applied_cras,
            self.depth + 1
        )
        return new_state

    def get_value(self) -> float:
        if self.terminal:
            if len(self.conflicts) == 0:
                total_cost = sum(cra.estimated_resolution_time for cra in self.applied_cras)
                return 1.0 - min(total_cost / 1000.0, 0.9)
            else:
                return -0.5 - 0.1 * len(self.conflicts)
        else:
            return 0.5 - 0.05 * len(self.conflicts)


class ConflictDetector:
    def __init__(self, schedule: pd.DataFrame):
        self.schedule = schedule

    def detect_all_conflicts(self) -> List[Dict]:
        conflicts = []
        node_conflicts = self.detect_node_conflicts()
        conflicts.extend(node_conflicts)
        link_conflicts = self.detect_link_conflicts()
        conflicts.extend(link_conflicts)
        return conflicts

    def detect_node_conflicts(self) -> List[Dict]:
        conflicts = []
        for node in self.schedule['NODE'].unique():
            node_schedule = self.schedule[self.schedule['NODE'] == node]
            trains = node_schedule['TRAIN_COURSE_ID'].unique()
            if len(trains) < 2:
                continue
            for i in range(len(trains)):
                for j in range(i + 1, len(trains)):
                    train1_data = node_schedule[node_schedule['TRAIN_COURSE_ID'] == trains[i]]
                    train2_data = node_schedule[node_schedule['TRAIN_COURSE_ID'] == trains[j]]
                    for _, row1 in train1_data.iterrows():
                        for _, row2 in train2_data.iterrows():
                            if pd.notna(row1.get('TRACK')) and row1.get('TRACK') == row2.get('TRACK'):
                                arr1 = row1['ARRIVAL_SECONDS'] if pd.notna(row1['ARRIVAL_SECONDS']) else row1['DEPARTURE_SECONDS'] - 30
                                dep1 = row1['DEPARTURE_SECONDS']
                                arr2 = row2['ARRIVAL_SECONDS'] if pd.notna(row2['ARRIVAL_SECONDS']) else row2['DEPARTURE_SECONDS'] - 30
                                dep2 = row2['DEPARTURE_SECONDS']
                                if not (dep1 <= arr2 or arr1 >= dep2):
                                    conflict = {
                                        'course_id_1': trains[i],
                                        'course_id_2': trains[j],
                                        'course_seq_1': row1['SEQ'],
                                        'course_seq_2': row2['SEQ'],
                                        'start_node': node,
                                        'end_node': node,
                                        'place': 'NODE',
                                        'detail': row1.get('TRACK', 'N/A'),
                                        'start_secs': max(arr1, arr2),
                                        'conflict_type': 'OC'
                                    }
                                    conflicts.append(conflict)
        return conflicts

    def detect_link_conflicts(self) -> List[Dict]:
        conflicts = []
        trains = self.schedule['TRAIN_COURSE_ID'].unique()
        for i in range(len(trains)):
            for j in range(i + 1, len(trains)):
                train1_schedule = self.schedule[self.schedule['TRAIN_COURSE_ID'] == trains[i]].sort_values('SEQ')
                train2_schedule = self.schedule[self.schedule['TRAIN_COURSE_ID'] == trains[j]].sort_values('SEQ')
                for k in range(len(train1_schedule) - 1):
                    for l in range(len(train2_schedule) - 1):
                        curr1 = train1_schedule.iloc[k]
                        next1 = train1_schedule.iloc[k + 1]
                        curr2 = train2_schedule.iloc[l]
                        next2 = train2_schedule.iloc[l + 1]
                        if ((curr1['NODE'] == curr2['NODE'] and next1['NODE'] == next2['NODE']) or (curr1['NODE'] == next2['NODE'] and next1['NODE'] == curr2['NODE'])):
                            start1 = curr1['DEPARTURE_SECONDS']
                            end1 = next1['ARRIVAL_SECONDS'] if pd.notna(next1['ARRIVAL_SECONDS']) else start1 + 180
                            start2 = curr2['DEPARTURE_SECONDS']
                            end2 = next2['ARRIVAL_SECONDS'] if pd.notna(next2['ARRIVAL_SECONDS']) else start2 + 180
                            if not (end1 <= start2 or start1 >= end2):
                                required_headway = self.get_required_headway(curr1['NODE'], next1['NODE'])
                                actual_headway = abs(start2 - end1) if start1 < start2 else abs(start1 - end2)
                                if actual_headway < required_headway:
                                    conflict = {
                                        'course_id_1': trains[i],
                                        'course_id_2': trains[j],
                                        'course_seq_1': curr1['SEQ'],
                                        'course_seq_2': curr2['SEQ'],
                                        'start_node': curr1['NODE'],
                                        'end_node': next1['NODE'],
                                        'place': 'LINK',
                                        'detail': '',
                                        'start_secs': max(start1, start2),
                                        'conflict_type': 'OC'
                                    }
                                    conflicts.append(conflict)
        return conflicts

    def get_required_headway(self, start_node: str, end_node: str) -> float:
        headway_data = HEADWAY[((HEADWAY['LINK_START_NODE'] == start_node) & (HEADWAY['LINK_END_NODE'] == end_node)) | ((HEADWAY['LINK_START_NODE'] == end_node) & (HEADWAY['LINK_END_NODE'] == start_node))]
        if not headway_data.empty:
            return headway_data['MINIMUM_HEADWAY_SECONDS'].max()
        return 90


class MCTSNode:
    def __init__(self, state: ConflictState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = state.get_legal_actions()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop()
        next_state = self.state.apply_action(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def update(self, value: float):
        self.visits += 1
        self.value += value


class MCTS:
    def __init__(self, neural_net=None, args=None):
        self.neural_net = neural_net
        self.args = args or {'num_mcts_sims': 100, 'cpuct': 1.0}

    def search(self, root_state: ConflictState) -> Dict[CRA, float]:
        root = MCTSNode(root_state)
        for _ in range(self.args['num_mcts_sims']):
            node = root
            while not node.state.terminal and node.is_fully_expanded():
                node = node.best_child(self.args['cpuct'])
            if not node.state.terminal and not node.is_fully_expanded():
                node = node.expand()
            if self.neural_net:
                value = self.neural_net.predict(node.state)[1]
            else:
                value = node.state.get_value()
            while node is not None:
                node.update(value)
                node = node.parent
        action_visits = {}
        for action, child in root.children.items():
            action_visits[action] = child.visits
        total_visits = sum(action_visits.values())
        action_probs = {a: v / total_visits for a, v in action_visits.items()}
        return action_probs

    def get_best_action(self, state: ConflictState) -> Optional[CRA]:
        if state.terminal:
            return None
        action_probs = self.search(state)
        if not action_probs:
            return None
        best_action = max(action_probs.items(), key=lambda x: x[1])[0]
        return best_action
