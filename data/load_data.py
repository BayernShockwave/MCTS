from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
ORIGIN_FILE = BASE_DIR / "origin" / "ELIZABETH_LINE_DATA.xlsx"

if not ORIGIN_FILE.exists():
    raise FileNotFoundError(f"FILE NOT FOUND ERROR: {ORIGIN_FILE.resolve()}")

ALL_DUTY_START_END = pd.read_excel(ORIGIN_FILE, sheet_name="ALL_DUTY_START_END")
BASE_STATION_VALUE = pd.read_excel(ORIGIN_FILE, sheet_name="BASE_STATION_VALUE")
HEADWAY = pd.read_excel(ORIGIN_FILE, sheet_name="HEADWAY")
LINK = pd.read_excel(ORIGIN_FILE, sheet_name="LINK")
MINIMUM_RUN_TIME = pd.read_excel(ORIGIN_FILE, sheet_name="MINIMUM_RUN_TIME")
NODE = pd.read_excel(ORIGIN_FILE, sheet_name="NODE")
PADTLL_WCHAPXR_TARGET_FREQUENCY = pd.read_excel(ORIGIN_FILE, sheet_name="PADTLL_WCHAPXR_TARGET_FREQUENCY")
ROLLING_STOCK_DUTY = pd.read_excel(ORIGIN_FILE, sheet_name="ROLLING_STOCK_DUTY")
TRAIN_HEADER = pd.read_excel(ORIGIN_FILE, sheet_name="TRAIN_HEADER")
TRAIN_SCHEDULE = pd.read_excel(ORIGIN_FILE, sheet_name="TRAIN_SCHEDULE")

EXTENDED_RUN_TIME = pd.read_excel(ORIGIN_FILE, sheet_name="EXTENDED_RUN_TIME")
LATE_DEPARTURE = pd.read_excel(ORIGIN_FILE, sheet_name="LATE_DEPARTURE")
REALIZED_SCHEDULE = pd.read_excel(ORIGIN_FILE, sheet_name="REALIZED_SCHEDULE")
STATION_EXTENDED_DWELL = pd.read_excel(ORIGIN_FILE, sheet_name="STATION_EXTENDED_DWELL")
TRAIN_EXTENDED_DWELL = pd.read_excel(ORIGIN_FILE, sheet_name="TRAIN_EXTENDED_DWELL")
