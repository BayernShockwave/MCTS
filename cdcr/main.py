from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
SCHEDULE_OUTPUT_PATH = BASE_DIR / "data" / "realized_schedule.xlsx"
CONFLICT_OUTPUT_PATH = BASE_DIR / "data" / "earliest_conflict.xlsx"

# 以earliest_conflict.xlsx中的一个随机冲突最为MCTS的根节点
earliest_conflicts = pd.read_excel(CONFLICT_OUTPUT_PATH)
if len(earliest_conflicts) > 0:
    ROOT_CONFLICT = earliest_conflicts.iloc[np.random.randint(0, len(earliest_conflicts))]
else:
    print("NO DETECTED CONFLICT!")
