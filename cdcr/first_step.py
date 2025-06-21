from data.load_data import HEADWAY, MINIMUM_RUN_TIME, ROLLING_STOCK_DUTY, TRAIN_HEADER, TRAIN_SCHEDULE
import random
import pandas as pd
from utils.time_operation import seconds_to_hhmmss
from pathlib import Path

# 随机某趟列车发车晚点10分钟
train_list = TRAIN_HEADER['TRAIN_COURSE_ID'].tolist()
random_train = random.choice(train_list)
LATE_DEPARTURE = pd.DataFrame([{'TRAIN_COURSE_ID': random_train, 'DEPARTURE_DELAY_SECONDS': 600}])


# 根据随机晚点更新REALIZED_SCHEDULE
def update_realized_schedule():
    for _, row in LATE_DEPARTURE.iterrows():
        train_id = row['TRAIN_COURSE_ID']
        delay_seconds = row['DEPARTURE_DELAY_SECONDS']
        records = REALIZED_SCHEDULE['TRAIN_COURSE_ID'] == train_id
        if records.any():
            # 更新到站时间(始发站除外)
            REALIZED_SCHEDULE.loc[records, 'ARRIVAL_SECONDS'] = \
            REALIZED_SCHEDULE.loc[records, 'ARRIVAL_SECONDS'].apply(lambda x: x + delay_seconds if pd.notna(x) else x)
            # 更新出站时间
            REALIZED_SCHEDULE.loc[records, 'DEPARTURE_SECONDS'] = \
            REALIZED_SCHEDULE.loc[records, 'DEPARTURE_SECONDS'] + delay_seconds
            # 更新HHMMSS
            for idx in REALIZED_SCHEDULE[records].index:
                if pd.notna(REALIZED_SCHEDULE.loc[idx, 'ARRIVAL_SECONDS']):
                    REALIZED_SCHEDULE.loc[idx, 'ARRIVAL_HHMMSS'] = seconds_to_hhmmss(REALIZED_SCHEDULE.loc[idx, 'ARRIVAL_SECONDS'])
                REALIZED_SCHEDULE.loc[idx, 'DEPARTURE_HHMMSS'] = seconds_to_hhmmss(REALIZED_SCHEDULE.loc[idx, 'DEPARTURE_SECONDS'])
            print(f"Train {train_id} has been rescheduled and is delayed by {delay_seconds} seconds.")


# 冲突检测
class ConflictDetector:
    def __init__(self):
        self.conflict_dict = {
            'course_id_1': [],
            'course_id_2': [],
            'course_seq_1': [],
            'course_seq_2': [],
            'duty_id_1': [],
            'duty_id_2': [],
            'duty_seq_1': [],
            'duty_seq_2': [],
            'interval_1': [],
            'interval_2': [],
            'start_node': [],
            'end_node': [],
            'place': [],  # NODE or LINK
            'detail': [],  # TRACK for NODE and '' for LINK
            'start_secs': [],
            'start_time': [],
            'time_gap': [],
            'conflict_type': []
        }

    def detect_all_conflicts(self):
        print("\nConflict Detection Start:")
        print(f"Delayed Train: {random_train}")
        self.detect_node_conflicts()
        self.detect_link_conflicts()
        return {
            'delayed_train': random_train,
            'conflict_dict': self.conflict_dict,
            'total_conflicts': len(self.conflict_dict['course_id_1'])
        }

    def detect_node_conflicts(self):
        delayed_train_schedule = REALIZED_SCHEDULE[
            REALIZED_SCHEDULE['TRAIN_COURSE_ID'] == random_train
        ]
        delayed_duty_info = self.get_duty_info(random_train)
        for _, row in delayed_train_schedule.iterrows():
            node = row['NODE']
            track = row.get('TRACK', None)
            seq = row['SEQ']
            # 占用窗口, 若没有进站时间, 则窗口从出站时间 - 30s开始
            arrival = row['ARRIVAL_SECONDS'] if pd.notna(row['ARRIVAL_SECONDS']) else row['DEPARTURE_SECONDS'] - 30
            departure = row['DEPARTURE_SECONDS']
            other_trains = REALIZED_SCHEDULE[
                (REALIZED_SCHEDULE['NODE'] == node) & (REALIZED_SCHEDULE['TRAIN_COURSE_ID'] != random_train)
            ]
            if pd.notna(track) and track != '':
                other_trains = other_trains[other_trains['TRACK'] == track]
            for _, other_row in other_trains.iterrows():
                other_arrival = other_row['ARRIVAL_SECONDS'] if pd.notna(other_row['ARRIVAL_SECONDS']) else other_row['DEPARTURE_SECONDS'] - 30
                other_departure = other_row['DEPARTURE_SECONDS']
                other_seq = other_row['SEQ']
                if not (departure <= other_arrival or arrival >= other_departure):
                    other_duty_info = self.get_duty_info(other_row['TRAIN_COURSE_ID'])
                    overlap_start = max(arrival, other_arrival)
                    overlap_end = min(departure, other_departure)
                    overlap_seconds = overlap_end - overlap_start
                    self.conflict_dict['course_id_1'].append(random_train)
                    self.conflict_dict['course_id_2'].append(other_row['TRAIN_COURSE_ID'])
                    self.conflict_dict['course_seq_1'].append(seq)
                    self.conflict_dict['course_seq_2'].append(other_seq)
                    self.conflict_dict['duty_id_1'].append(delayed_duty_info['duty_id'])
                    self.conflict_dict['duty_id_2'].append(other_duty_info['duty_id'])
                    self.conflict_dict['duty_seq_1'].append(delayed_duty_info['seq'])
                    self.conflict_dict['duty_seq_2'].append(other_duty_info['seq'])
                    self.conflict_dict['interval_1'].append((arrival, departure))
                    self.conflict_dict['interval_2'].append((other_arrival, other_departure))
                    self.conflict_dict['start_node'].append(node)
                    self.conflict_dict['end_node'].append(node)
                    self.conflict_dict['place'].append('NODE')
                    self.conflict_dict['detail'].append(track if pd.notna(track) else 'N/A')
                    self.conflict_dict['start_secs'].append(overlap_start)
                    self.conflict_dict['start_time'].append(seconds_to_hhmmss(overlap_start))
                    self.conflict_dict['time_gap'].append(overlap_seconds)
                    self.conflict_dict['conflict_type'].append('OC')

    def detect_link_conflicts(self):
        delayed_train_schedule = REALIZED_SCHEDULE[
            REALIZED_SCHEDULE['TRAIN_COURSE_ID'] == random_train
        ].sort_values('SEQ')
        delayed_duty_info = self.get_duty_info(random_train)
        for i in range(len(delayed_train_schedule) - 1):
            current_stop = delayed_train_schedule.iloc[i]
            next_stop = delayed_train_schedule.iloc[i + 1]
            start_node = current_stop['NODE']
            end_node = next_stop['NODE']
            link_start_time = current_stop['DEPARTURE_SECONDS']
            link_end_time = next_stop['ARRIVAL_SECONDS']
            current_seq = current_stop['SEQ']
            # 若没有进站时间, 则使用最小运行时间估算
            if pd.isna(link_end_time):
                min_run_time = self.get_minimum_run_time(start_node, end_node)
                link_end_time = link_start_time + min_run_time
            for train_id in REALIZED_SCHEDULE['TRAIN_COURSE_ID'].unique():
                if train_id == random_train:
                    continue
                others_schedule = REALIZED_SCHEDULE[
                    REALIZED_SCHEDULE['TRAIN_COURSE_ID'] == train_id
                ]
                for j in range(len(others_schedule) - 1):
                    current = others_schedule.iloc[j]
                    next = others_schedule.iloc[j + 1]
                    if ((current['NODE'] == start_node and next['NODE'] == end_node) or (current['NODE'] == end_node and next['NODE'] == start_node)):
                        other_start = current['DEPARTURE_SECONDS']
                        other_end = next['ARRIVAL_SECONDS']
                        other_seq = current['SEQ']
                        if pd.isna(other_end):
                            min_run_time = self.get_minimum_run_time(current['NODE'], next['NODE'])
                            other_end = other_start + min_run_time
                        if not (link_end_time <= other_start or link_start_time >= other_end):
                            # 其它列车在前
                            if other_start < link_start_time:
                                headway = link_start_time - other_end
                                conflict_start = other_end
                            # 延误列车在前
                            else:
                                headway = other_start - link_end_time
                                conflict_start = link_end_time
                            required_headway = self.get_required_headway(start_node, end_node)
                            if headway < required_headway:
                                other_duty_info = self.get_duty_info(train_id)
                                self.conflict_dict['course_id_1'].append(random_train)
                                self.conflict_dict['course_id_2'].append(train_id)
                                self.conflict_dict['course_seq_1'].append(current_seq)
                                self.conflict_dict['course_seq_2'].append(other_seq)
                                self.conflict_dict['duty_id_1'].append(delayed_duty_info['duty_id'])
                                self.conflict_dict['duty_id_2'].append(other_duty_info['duty_id'])
                                self.conflict_dict['duty_seq_1'].append(delayed_duty_info['seq'])
                                self.conflict_dict['duty_seq_2'].append(other_duty_info['seq'])
                                self.conflict_dict['interval_1'].append((link_start_time, link_end_time))
                                self.conflict_dict['interval_2'].append((other_start, other_end))
                                self.conflict_dict['start_node'].append(start_node)
                                self.conflict_dict['end_node'].append(end_node)
                                self.conflict_dict['place'].append('LINK')
                                self.conflict_dict['detail'].append('')
                                self.conflict_dict['start_secs'].append(conflict_start)
                                self.conflict_dict['start_time'].append(seconds_to_hhmmss(conflict_start))
                                self.conflict_dict['time_gap'].append(required_headway - max(0, headway))
                                self.conflict_dict['conflict_type'].append('OC')
                            break

    def get_duty_info(self, train_course_id):
        duty_info = ROLLING_STOCK_DUTY[
            ROLLING_STOCK_DUTY['TRAIN_COURSE_ID'] == train_course_id
        ]
        if not duty_info.empty:
            return {
                'duty_id': duty_info.iloc[0]['DUTY_ID'],
                'seq': duty_info.iloc[0]['SEQ']
            }
        return {
            'duty_id': 'N/A',
            'seq': 0
        }

    def get_minimum_run_time(self, start_node, end_node):
        min_time = MINIMUM_RUN_TIME[
            (MINIMUM_RUN_TIME['LINK_START_NODE'] == start_node) & (MINIMUM_RUN_TIME['LINK_END_NODE'] == end_node)
        ]
        if not min_time.empty:
            return min_time['MINIMUM_RUN_TIME_SECONDS'].max()
        return 180

    def get_required_headway(self, start_node, end_node):
        headway_data = HEADWAY[((HEADWAY['LINK_START_NODE'] == start_node) & (HEADWAY['LINK_END_NODE'] == end_node)) | ((HEADWAY['LINK_START_NODE'] == end_node) & (HEADWAY['LINK_END_NODE'] == start_node))]
        if not headway_data.empty:
            return headway_data['MINIMUM_HEADWAY_SECONDS'].max()
        return 90


BASE_DIR = Path(__file__).parent
SCHEDULE_OUTPUT_PATH = BASE_DIR / "data" / "realized_schedule.xlsx"
CONFLICT_OUTPUT_PATH = BASE_DIR / "data" / "earliest_conflict.xlsx"

column_mapping = {
    'TRAIN_COURSE_ID': 'TRAIN_COURSE_ID',
    'SEQ': 'SEQ',
    'NODE': 'NODE',
    'ARRIVAL_SECONDS': 'ARRIVAL_SECONDS',
    'ARRIVAL_HHMMSS': 'ARRIVAL_HHMMSS',
    'DEPARTURE_SECONDS': 'DEPARTURE_SECONDS',
    'DEPARTURE_HHMMSS': 'DEPARTURE_HHMMSS',
    'Track': 'TRACK'
}
REALIZED_SCHEDULE = TRAIN_SCHEDULE[list(column_mapping.keys())].rename(columns=column_mapping)

update_realized_schedule()
REALIZED_SCHEDULE.to_excel(SCHEDULE_OUTPUT_PATH, index=False)

detector = ConflictDetector()
conflicts = detector.detect_all_conflicts()
print(f"\nConflict Detection Complete:")
print(f"Total Conflicts Number: {conflicts['total_conflicts']}")
conflict_df = pd.DataFrame(conflicts['conflict_dict'])
if not conflict_df.empty:
    print(conflict_df)
    conflict_df = conflict_df.sort_values('start_secs')
    earliest_start = conflict_df['start_secs'].min()
    earliest_conflict_df = conflict_df[conflict_df['start_secs'] == earliest_start]
    earliest_conflict_df.to_excel(CONFLICT_OUTPUT_PATH, index=False)
