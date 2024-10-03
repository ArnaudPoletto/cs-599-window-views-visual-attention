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
DEPTH_PATH = f"{DATA_PATH}/depth"
DEPTH_MAP_IMG_PATH = f"{DEPTH_PATH}/map_images"
DEPTH_MAP_PFM_PATH = f"{DEPTH_PATH}/map_pfm"
DEPTH_SEG_IMG_PATH = f"{DEPTH_PATH}/segmented_images"
DEPTH_SEG_PFM_PATH = f"{DEPTH_PATH}/segmented_pfm"
SALIENCY_PATH = f"{DATA_PATH}/saliency"
SALIENCY_MAP_IMG_PATH = f"{SALIENCY_PATH}/map_images"
SALIENCY_MAP_PFM_PATH = f"{SALIENCY_PATH}/map_pfm"

RAW_EYE_TRACKING_FRAME_WIDTH = 6144
RAW_EYE_TRACKING_FRAME_HEIGHT = 3072