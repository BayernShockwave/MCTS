import pandas as pd
from typing import List, Dict, Optional
from utils.classes import CRA
from data.load_data import NODE, MINIMUM_RUN_TIME, HEADWAY
from utils.time_operation import seconds_to_hhmmss
import numpy as np


class ConflictState:
    def __init__(self, realized_schedule: pd.DataFrame, conflicts: List[Dict], applied_cras: List[CRA] = None, depth: int = 0, time_horizon: float = None):
        self.realized_schedule = realized_schedule.copy()
        self.conflicts = conflicts
        self.applied_cras = applied_cras if applied_cras is not None else []
        self.depth = depth
        # 仅关注30分钟内的冲突
        if time_horizon is None and conflicts:
            self.time_horizon = min(c['start_secs'] for c in conflicts) + 1800
        else:
            self.time_horizon = time_horizon
        self.terminal = len(conflicts) == 0 or depth >= 10
        self.legal_actions_cache = None

    def get_legal_actions(self) -> List[CRA]:
        if self.legal_actions_cache is not None:
            return self.legal_actions_cache
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
                node_info = NODE[NODE['CODE'] == conflict['start_node']]
                available_tracks = []
                for _, row in node_info.iterrows():
                    if row['EB_TRACKS'] and row['EB_TRACKS'] != conflict['detail']:
                        available_tracks.append(row['EB_TRACKS'])
                    if row['WB_TRACKS'] and row['WB_TRACKS'] != conflict['detail']:
                        available_tracks.append(row['WB_TRACKS'])
                for train_id in [conflict['course_id_1'], conflict['course_id_2']]:
                    for track in available_tracks:
                        resolution_time = self.calculate_track_change_time(conflict['detail'], track)
                        cra = CRA(
                            ecs_list=['RRT'],
                            target_trains=[train_id],
                            target_locations=[conflict['start_node']],
                            parameters={'new_track': track},
                            conflict_index=idx,
                            estimated_resolution_time=resolution_time
                        )
                        actions.append(cra)
        self.legal_actions_cache = actions
        return actions

    def calculate_track_change_time(self, current_track: str, new_track: str) -> float:
        current = ''.join(filter(str.isdigit, str(current_track)))
        new = ''.join(filter(str.isdigit, str(new_track)))
        if current and new:
            try:
                current_num = int(current)
                new_num = int(new)
                track_diff = abs(new_num - current_num)
                return 180 + (track_diff - 1) * 120 if track_diff > 0 else 0
            except ValueError:
                pass
        return 180

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
        detector = ConflictDetector(new_schedule, self.time_horizon)
        new_conflicts = detector.detect_all_conflicts()
        new_applied_cras = self.applied_cras + [cra]
        new_state = ConflictState(
            new_schedule,
            new_conflicts,
            new_applied_cras,
            self.depth + 1,
            self.time_horizon
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
            conflict_penalty = 0.05 * len(self.conflicts)
            depth_penalty = 0.02 * self.depth
            return 0.5 - conflict_penalty - depth_penalty


class ConflictDetector:
    def __init__(self, schedule: pd.DataFrame, time_horizon: float = None):
        self.schedule = schedule
        self.time_horizon = time_horizon
        self.train_groups = schedule.groupby('TRAIN_COURSE_ID')
        self.node_groups = schedule.groupby('NODE')

    def detect_all_conflicts(self) -> List[Dict]:
        conflicts = []
        node_conflicts = self.detect_node_conflicts()
        conflicts.extend(node_conflicts)
        link_conflicts = self.detect_link_conflicts()
        conflicts.extend(link_conflicts)
        return conflicts

    def detect_node_conflicts(self) -> List[Dict]:
        conflicts = []
        for node, node_data in self.node_groups:
            track_groups = node_data.groupby('TRACK')
            for track, track_data in track_groups:
                if pd.isna(track) or track == '':
                    continue
                trains = track_data['TRAIN_COURSE_ID'].unique()
                if len(trains) < 2:
                    continue
                for i in range(len(trains)):
                    for j in range(i + 1, len(trains)):
                        train1_data = track_data[track_data['TRAIN_COURSE_ID'] == trains[i]]
                        train2_data = track_data[track_data['TRAIN_COURSE_ID'] == trains[j]]
                        for _, row1 in train1_data.iterrows():
                            for _, row2 in train2_data.iterrows():
                                arr1 = row1['ARRIVAL_SECONDS'] if pd.notna(row1['ARRIVAL_SECONDS']) else row1['DEPARTURE_SECONDS'] - 30
                                dep1 = row1['DEPARTURE_SECONDS']
                                arr2 = row2['ARRIVAL_SECONDS'] if pd.notna(row2['ARRIVAL_SECONDS']) else row2['DEPARTURE_SECONDS'] - 30
                                dep2 = row2['DEPARTURE_SECONDS']
                                if not (dep1 <= arr2 or arr1 >= dep2):
                                    overlap_start = max(arr1, arr2)
                                    overlap_end = min(dep1, dep2)
                                    if self.time_horizon is not None and overlap_start > self.time_horizon:
                                        continue
                                    conflict = {
                                        'course_id_1': trains[i],
                                        'course_id_2': trains[j],
                                        'course_seq_1': row1['SEQ'],
                                        'course_seq_2': row2['SEQ'],
                                        'start_node': node,
                                        'end_node': node,
                                        'place': 'NODE',
                                        'detail': track,
                                        'start_secs': overlap_start,
                                        'start_time': seconds_to_hhmmss(overlap_start),
                                        'time_gap': overlap_end - overlap_start,
                                        'conflict_type': 'OC'
                                    }
                                    conflicts.append(conflict)
        return conflicts

    def detect_link_conflicts(self) -> List[Dict]:
        conflicts = []
        link_occupations = []
        for train_id, train_data in self.train_groups:
            train_schedule = train_data.sort_values('SEQ')
            for i in range(len(train_schedule) - 1):
                current = train_schedule.iloc[i]
                next_stop = train_schedule.iloc[i + 1]
                start_time = current['DEPARTURE_SECONDS']
                end_time = next_stop['ARRIVAL_SECONDS']
                if pd.isna(end_time):
                    min_time = self.get_minimum_run_time(current['NODE'], next_stop['NODE'])
                    end_time = start_time + min_time
                if self.time_horizon is not None and start_time > self.time_horizon:
                    continue
                link_occupations.append({
                    'train_id': train_id,
                    'start_node': current['NODE'],
                    'end_node': next_stop['NODE'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'seq': current['SEQ']
                })
        for i in range(len(link_occupations)):
            for j in range(i + 1, len(link_occupations)):
                occ1 = link_occupations[i]
                occ2 = link_occupations[j]
                same_link = ((occ1['start_node'] == occ2['start_node'] and occ1['end_node'] == occ2['end_node']) or (occ1['start_node'] == occ2['end_node'] and occ1['end_node'] == occ2['start_node']))
                if not same_link:
                    continue
                if not (occ1['end_time'] <= occ2['start_time'] or occ1['start_time'] >= occ2['end_time']):
                    if occ1['start_time'] < occ2['start_time']:
                        headway = occ2['start_time'] - occ1['end_time']
                        conflict_start = occ1['end_time']
                    else:
                        headway = occ1['start_time'] - occ2['end_time']
                        conflict_start = occ2['end_time']
                    required_headway = self.get_required_headway(occ1['start_node'], occ1['end_node'])
                    if headway < required_headway:
                        if self.time_horizon is not None and conflict_start > self.time_horizon:
                            continue
                        conflict = {
                            'course_id_1': occ1['train_id'],
                            'course_id_2': occ2['train_id'],
                            'course_seq_1': occ1['seq'],
                            'course_seq_2': occ2['seq'],
                            'start_node': occ1['start_node'],
                            'end_node': occ1['end_node'],
                            'place': 'LINK',
                            'detail': '',
                            'start_secs': conflict_start,
                            'start_time': seconds_to_hhmmss(conflict_start),
                            'time_gap': required_headway - max(0, headway),
                            'conflict_type': 'OC'
                        }
                        conflicts.append(conflict)
        return conflicts

    def get_minimum_run_time(self, start_node: str, end_node: str) -> float:
        min_time = MINIMUM_RUN_TIME[
            (MINIMUM_RUN_TIME['LINK_START_NODE'] == start_node) &
            (MINIMUM_RUN_TIME['LINK_END_NODE'] == end_node)
            ]
        if not min_time.empty:
            return min_time['MINIMUM_RUN_TIME_SECONDS'].max()
        return 180

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
        self.nn_evaluated = False

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        choices_weights = []
        children_list = list(self.children.values())
        for child in children_list:
            avg_value = child.value / child.visits
            exploration = c_param * np.sqrt((2 * np.log(self.visits)) / child.visits)
            choices_weights.append(avg_value + exploration)
        best_idx = np.argmax(choices_weights)
        return children_list[best_idx]

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop()
        next_state = self.state.apply_action(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def update(self, value: float):
        self.visits += 1
        self.value += value

    def get_best_action(self) -> Optional[CRA]:
        if not self.children:
            return None
        return max(self.children.items(), key=lambda x: x[1].visits)[0]


class MCTS:
    def __init__(self, neural_net=None, args=None):
        self.neural_net = neural_net
        self.args = args or {'num_mcts_sims': 100, 'cpuct': 1.0}
        self.transposition_table = {}

    def search(self, root_state: ConflictState) -> Dict[CRA, float]:
        root = MCTSNode(root_state)
        for _ in range(self.args['num_mcts_sims']):
            node = root
            while not node.state.terminal and node.is_fully_expanded():
                node = node.best_child(self.args['cpuct'])
            if not node.state.terminal and not node.is_fully_expanded():
                node = node.expand()
            if self.neural_net and not node.nn_evaluated:
                _, value = self.neural_net.predict(node.state)
                node.nn_evaluated = True
            else:
                value = self.rollout(node.state)
            while node is not None:
                node.update(value)
                node = node.parent
        action_visits = {}
        for action, child in root.children.items():
            action_visits[action] = child.visits
        total_visits = sum(action_visits.values())
        if total_visits == 0:
            return {}
        action_probs = {a: v / total_visits for a, v in action_visits.items()}
        return action_probs

    def rollout(self, state: ConflictState) -> float:
        current_state = state
        while not current_state.terminal:
            actions = current_state.get_legal_actions()
            if not actions:
                break
            action = np.random.choice(actions)
            current_state = current_state.apply_action(action)
        return current_state.get_value()

    def get_best_action(self, state: ConflictState) -> Optional[CRA]:
        if state.terminal:
            return None
        action_probs = self.search(state)
        if not action_probs:
            return None
        best_action = max(action_probs.items(), key=lambda x: x[1])[0]
        return best_action
