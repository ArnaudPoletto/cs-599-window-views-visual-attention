import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from justpfm import justpfm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.utils.file import get_files_recursive, get_ids_from_file_path, get_set_str
from src.config import (
    IMAGES_PATH,
    DEPTH_MAP_IMG_PATH,
    DEPTH_MAP_PFM_PATH,
)

MODEL_SIZE = "Base"
MAX_DEPTH_VALUE = 5


def main() -> None:
    """
    Main function for estimating depth maps from monocular images using the Depth Anything V2 model.
    """
    # Load model
    print(f"🔃 Loading Depth Anything V2 model, using {MODEL_SIZE} model.")
    image_processor = AutoImageProcessor.from_pretrained(
        f"depth-anything/Depth-Anything-V2-{MODEL_SIZE}-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        f"depth-anything/Depth-Anything-V2-{MODEL_SIZE}-hf"
    )

    image_file_paths = get_files_recursive(IMAGES_PATH, "*.png")
    for image_file_path in tqdm(
        image_file_paths, desc="⌛ Computing depth maps...", unit="image"
    ):
        # Load image and perform inference
        image = cv2.imread(image_file_path)
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize depth map to original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        prediction = torch.clamp(prediction, min=0)
        depth_map = prediction.squeeze().cpu().numpy()

        # Save depth map
        experiment_id, set_id, sequence_id = get_ids_from_file_path(image_file_path)
        set_str = get_set_str(experiment_id, set_id)
        depth_map_pfm_path = f"{DEPTH_MAP_PFM_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}.pfm"
        os.makedirs(os.path.dirname(depth_map_pfm_path), exist_ok=True)
        justpfm.write_pfm(file_name=depth_map_pfm_path, data=depth_map)

        depth_map_img_path = f"{DEPTH_MAP_IMG_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}.png"
        depth_map = np.clip(depth_map, 0, MAX_DEPTH_VALUE)
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        os.makedirs(os.path.dirname(depth_map_img_path), exist_ok=True)
        cv2.imwrite(depth_map_img_path, depth_map)

    print(f"✅ Depth maps computed and saved at {Path(DEPTH_MAP_PFM_PATH).resolve()}.")


if __name__ == "__main__":
    main()
