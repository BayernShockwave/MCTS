from pathlib import Path
import pandas as pd
import numpy as np
from cdcr.cdcr_mcts import ConflictState, MCTS
from cdcr.neural_network import NeuralNetWrapper
from cdcr.self_play import Coach


def main():
    BASE_DIR = Path(__file__).parent
    SCHEDULE_OUTPUT_PATH = BASE_DIR / "data" / "realized_schedule.xlsx"
    CONFLICT_OUTPUT_PATH = BASE_DIR / "data" / "earliest_conflict.xlsx"
    realized_schedule = pd.read_excel(SCHEDULE_OUTPUT_PATH)
    earliest_conflicts = pd.read_excel(CONFLICT_OUTPUT_PATH)
    if len(earliest_conflicts) == 0:
        print("NO DETECTED CONFLICT!")
        return
        # 以earliest_conflict.xlsx中的一个随机冲突最为MCTS的根节点
        ROOT_CONFLICT = earliest_conflicts.iloc[np.random.randint(0, len(earliest_conflicts))]
        print(f"\nChosen Root:")
        print(f"Train 1: {ROOT_CONFLICT['course_id_1']} (SEQ: {ROOT_CONFLICT['course_seq_1']})")
        print(f"Train 2: {ROOT_CONFLICT['course_id_2']} (SEQ: {ROOT_CONFLICT['course_seq_2']})")
        print(f"Place: {ROOT_CONFLICT['place']} at {ROOT_CONFLICT['start_node']}")
        print(f"Start time: {ROOT_CONFLICT['start_time']}")
        print(f"Type: {ROOT_CONFLICT['conflict_type']}")
        initial_conflicts = []
        for _, row in earliest_conflicts.iterrows():
            conflict = {
                'course_id_1': row['course_id_1'],
                'course_id_2': row['course_id_2'],
                'course_seq_1': row['course_seq_1'],
                'course_seq_2': row['course_seq_2'],
                'duty_id_1': row.get('duty_id_1', 'N/A'),
                'duty_id_2': row.get('duty_id_2', 'N/A'),
                'duty_seq_1': row.get('duty_seq_1', 0),
                'duty_seq_2': row.get('duty_seq_2', 0),
                'interval_1': eval(row['interval_1']) if isinstance(row['interval_1'], str) else row['interval_1'],
                'interval_2': eval(row['interval_2']) if isinstance(row['interval_2'], str) else row['interval_2'],
                'start_node': row['start_node'],
                'end_node': row['end_node'],
                'place': row['place'],
                'detail': row['detail'],
                'start_secs': row['start_secs'],
                'start_time': row['start_time'],
                'time_gap': row['time_gap'],
                'conflict_type': row['conflict_type']
            }
            initial_conflicts.append(conflict)
        initial_state = ConflictState(
            realized_schedule=realized_schedule,
            conflicts=initial_conflicts,
            applied_cras=[],
            depth=0
        )
        args = {
            'numIters': 100,
            'numEps': 50,
            'tempThreshold': 15,
            'updateThreshold': 0.55,
            'maxlenOfQueue': 200000,
            'num_mcts_sims': 50,
            'arenaCompare': True,
            'arenaCompareGames': 20,
            'cpuct': 1.0,
            'checkpoint': './checkpoint',
            'load_model': False,
            'load_folder_file': ('./checkpoint/', 'best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,
            'epochs': 10,
            'batch_size': 64,
            'valueDecay': 0.99,
            'verbose': True
        }
        checkpoint_dir = Path(args['checkpoint'])
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)
        nnet = NeuralNetWrapper(input_size=512, hidden_size=256, action_size=1000)
        if args['load_model']:
            model_path = Path(args['load_folder_file'][0]) / args['load_folder_file'][1]
            if model_path.exists():
                print('Loading checkpoint...')
                nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
            else:
                print(f"Warning: Model file {model_path} not found. Starting fresh.")
                args['load_model'] = False
        coach = Coach(initial_state, nnet, args)
        if args['load_model']:
            print("Loading training examples...")
            coach.loadTrainExamples()
        print('\nStarting self-play learning...')
        print(f"Total conflicts to resolve: {len(initial_conflicts)}")
        print("\nStarting Neural Network Training...")
        coach.learn()
        print("\nSolution with Trained Neural Network:")
        trained_mcts = MCTS(nnet, args)
        final_state = initial_state
        final_actions = []
        for step in range(10):
            if final_state.terminal or len(final_state.conflicts) == 0:
                break
            action = trained_mcts.get_best_action(final_state)
            if action:
                final_actions.append(action)
                final_state = final_state.apply_action(action)
                print(f"Step {step + 1}: Applied {action.ecs_list} to {action.target_trains}")
                print(f"Parameters: {action.parameters}")
                print(f"Remaining conflicts: {len(final_state.conflicts)}")
        print("\nFINAL RESULTS:")
        print(f"Initial conflicts: {len(initial_conflicts)}")
        print(f"Remaining conflicts: {len(final_state.conflicts)}")
        print(f"Actions taken: {len(final_actions)}")
        print(f"Total resolution time: {sum(a.estimated_resolution_time for a in final_actions)} seconds")


if __name__ == "__main__":
    main()
