import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

GENERATED_PATH = str(GLOBAL_DIR / "generated")
DATA_PATH = str(GLOBAL_DIR / "data")
RAW_EYE_TRACKING_DATA_PATH = f"{DATA_PATH}/eye_tracking/raw"
PROCESSED_EYE_TRACKING_DATA_PATH = f"{DATA_PATH}/eye_tracking/processed"
PROCESSED_EYE_TRACKING_FILE_NAME = "data.csv"
SETS_PATH = f"{DATA_PATH}/sets"

RAW_EYE_TRACKING_FRAME_WIDTH = 6144
RAW_EYE_TRACKING_FRAME_HEIGHT = 3072