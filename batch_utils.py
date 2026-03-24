import concurrent.futures
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image


# =========================
# THREAD EXECUTOR
# =========================

def create_batch_executor(max_workers: int = 4):
    """Create thread pool for parallel image processing"""
    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


# =========================
# IMAGE VALIDATION
# =========================

def validate_image(input_path: str, max_size_mb: int = 20) -> Tuple[bool, str]:
    """
    Validate image by extension, size, existence, and loadability.
    """

    try:
        path = Path(input_path)

        if not path.exists():
            return False, "File not found"

        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        ext = path.suffix.lower()

        if ext not in allowed_extensions:
            return False, "Unsupported format (JPG/JPEG/PNG/WEBP only)"

        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb > max_size_mb:
            return False, f"File too large ({size_mb:.1f} MB > {max_size_mb} MB)"

        img = cv2.imread(str(path))

        if img is None:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return False, "Failed to load image"

        if img is None or img.size == 0:
            return False, "Invalid or corrupt image"

        return True, ""

    except Exception as e:
        return False, str(e)


# =========================
# METRICS
# =========================

def compute_metrics(original_path: str, enhanced_path: str) -> Dict[str, float]:
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)

    if original is None or enhanced is None:
        raise ValueError("Could not load one or both images.")

    # Make sure sizes match
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))

    # PSNR
    psnr_value = cv2.PSNR(original, enhanced)

    # SSIM
    ssim_value = ssim(original, enhanced, channel_axis=2, data_range=255)

    return {
        "psnr": round(psnr_value, 2),
        "ssim": round(ssim_value, 4)
    }


# =========================
# ZIP CREATOR
# =========================

def create_zip_batch(
    batch_id: str,
    output_paths: List[Path],
    original_names: List[str]
) -> str:
    """
    Create ZIP of enhanced batch results
    """

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    zip_path = output_dir / f"{batch_id}_batch_results.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zf:

        for out_path, orig_name in zip(output_paths, original_names):

            arcname = f"{Path(orig_name).stem}_enhanced.png"

            zf.write(out_path, arcname)

    return str(zip_path)