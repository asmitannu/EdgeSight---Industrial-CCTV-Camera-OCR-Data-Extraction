"""
Synthetic industrial label and CCTV image generator.
Creates realistic test images containing equipment nameplates, serial tags,
ratings, angles, and noise for pipeline verification and testing.
"""

import os
import cv2
import numpy as np


def create_motor_nameplate() -> np.ndarray:
    """Generates a clean industrial motor nameplate with structured text."""
    # 800 x 500 metallic plate
    img = np.full((500, 800, 3), 235, dtype=np.uint8)

    # Outer border & header banner
    cv2.rectangle(img, (20, 20), (780, 480), (80, 80, 80), 3)
    cv2.rectangle(img, (20, 20), (780, 90), (40, 40, 40), cv2.FILLED)

    # Header text
    cv2.putText(img, "SIEMENS SIMOTICS 3~ MOT", (40, 65), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # Equipment specifications
    cv2.putText(img, "MODEL: 1LE1001-1DB43-4AA4", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "SERIAL NO: SN-98420194-01", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "VOLTAGE: 400V / 690V  50Hz", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "POWER: 15.0 kW", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "SPEED: 1475 RPM", (400, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "ENCLOSURE: IP55  EFF: 93.6%", (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "DATE: 2026-03-15  TAG: MTR-PUMP-02", (40, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Barcode representation
    cv2.rectangle(img, (40, 420), (350, 460), (0, 0, 0), cv2.FILLED)
    for i in range(45, 345, 8):
        cv2.line(img, (i, 420), (i, 460), (255, 255, 255), 2)

    return img


def create_tilted_tag() -> np.ndarray:
    """Generates an equipment tag photographed at an angle (skewed)."""
    plate = create_motor_nameplate()

    # Resize slightly
    plate = cv2.resize(plate, (600, 380))

    # Canvas of industrial machinery background
    bg = np.full((700, 900, 3), 120, dtype=np.uint8)

    # Add background texture
    noise = np.random.normal(0, 15, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Rotate plate by -12 degrees
    h, w = plate.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, -12, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (nw / 2) - center[0]
    rot_mat[1, 2] += (nh / 2) - center[1]

    rotated_plate = cv2.warpAffine(plate, rot_mat, (nw, nh), borderValue=(120, 120, 120))

    # Place onto background canvas
    x_off = 100
    y_off = 120
    bg[y_off:y_off+nh, x_off:x_off+nw] = rotated_plate

    return bg


def create_low_contrast_cctv_image() -> np.ndarray:
    """Generates a low-contrast, shadowed CCTV equipment capture."""
    plate = create_motor_nameplate()

    # Darken and reduce contrast
    darkened = (plate * 0.35 + 20).astype(np.uint8)

    # Add Gaussian noise
    noise = np.random.normal(0, 10, darkened.shape).astype(np.int16)
    noisy = np.clip(darkened.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy


def generate_all_samples(output_dir: str = "input_images") -> None:
    """Generates and writes test sample images to disk."""
    os.makedirs(output_dir, exist_ok=True)

    samples = {
        "equipment_motor_01.png": create_motor_nameplate(),
        "equipment_tilted_tag_02.png": create_tilted_tag(),
        "equipment_cctv_shadow_03.png": create_low_contrast_cctv_image(),
    }

    for fname, img in samples.items():
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, img)
        print(f"Generated sample image: {fpath} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    generate_all_samples()
