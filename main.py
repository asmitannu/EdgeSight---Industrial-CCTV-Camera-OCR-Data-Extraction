"""
Main entry point for Stage 1 Industrial OCR Data Extraction System.
Processes digital photographs and CCTV captures of industrial equipment, applies
intelligent preprocessing variants, runs PaddleOCR, and exports structured CSV/Excel files.

Usage:
    python main.py --input input_images/
    python main.py --input input_images/sample.jpg --debug
    python main.py --input input_images/ --confidence 0.60 --roi 50,100,500,400
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import PipelineConfig, default_config
from src.image_loader import ImageLoader
from src.preprocessing import PreprocessingPipeline
from src.ocr_engine import PaddleOCREngine
from src.postprocessing import TextPostprocessor
from src.data_formatter import DataFormatter
from src.exporter import DataExporter


def setup_logging(debug_mode: bool = False) -> None:
    """Configures console and file logging."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def draw_debug_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Draws bounding boxes and text labels onto image for visual inspection.
    """
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for det in detections:
        bbox = det.get("bounding_box", [])
        text = det.get("text", "")
        conf = det.get("confidence", 0.0)

        if len(bbox) == 4:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw text label background
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))

            label = f"{text} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_ymin = max(y_min - 5, th + 5)

            cv2.rectangle(
                annotated,
                (x_min, label_ymin - th - 4),
                (x_min + tw + 4, label_ymin + baseline),
                (0, 0, 0),
                cv2.FILLED
            )
            cv2.putText(
                annotated,
                label,
                (x_min + 2, label_ymin - 2),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)


def process_single_image(
    image_path: str,
    config: PipelineConfig,
    ocr_engine: PaddleOCREngine,
    preprocessor: PreprocessingPipeline,
    roi_coords: Optional[tuple] = None
) -> List[Dict[str, Any]]:
    """
    Executes the end-to-end OCR extraction pipeline for a single image.

    Pipeline Steps:
        1. Load & validate image
        2. Optional ROI crop
        3. Generate preprocessing variants (CLAHE, Adaptive, Otsu, Deskew, Perspective, Upscaled)
        4. (Debug) Save variant images
        5. Run OCR on all variants
        6. Select best variant via composite scoring
        7. Postprocess extracted text
        8. (Debug) Save detection overlay image
        9. Return structured records
    """
    filename = os.path.basename(image_path)
    file_stem = os.path.splitext(filename)[0]

    print(f"\nProcessing image: {filename}")

    # 1. Load Image
    try:
        image = ImageLoader.load_image(image_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load {filename}: {e}")
        return []

    # 2. ROI Cropping
    if roi_coords is not None:
        try:
            image = ImageLoader.crop_roi(image, roi_coords)
            print(f"  Applied ROI crop: {roi_coords}")
        except Exception as e:
            print(f"  [WARNING] ROI crop failed: {e}. Processing full image.")

    # 3. Generate Preprocessing Variants
    print("Generating preprocessing variants...")
    variants = preprocessor.generate_variants(image, base_name=file_stem)

    # 4. Save debug variant images
    if config.debug_mode:
        preprocessor.save_variants(variants, config.processed_images_dir, file_stem)

    # 5. Run OCR on all variants
    print("Running OCR...")
    variant_results = {}
    for v_name, v_img in variants.items():
        ocr_res = ocr_engine.extract_text(v_img, image_name=filename, variant_name=v_name)
        variant_results[v_name] = ocr_res

    # 6. Multi-variant evaluation and selection
    best_raw_result = DataFormatter.select_best_variant(variant_results, config.scoring)
    winning_variant = best_raw_result.get("variant_name", "original")

    # 7. Postprocess text
    best_result = TextPostprocessor.postprocess_detections(best_raw_result)
    num_detections = len(best_result.get("detections", []))
    print(f"OCR completed. (Selected variant: '{winning_variant}', Detections: {num_detections})")

    # Display detections on console
    for det in best_result.get("detections", []):
        print(f"  -> '{det['text']}' (confidence: {det['confidence']:.2f})")

    # 8. Save debug annotated image
    if config.debug_mode and num_detections > 0:
        winning_img = variants.get(winning_variant, image)
        debug_annotated_path = os.path.join(config.debug_output_dir, f"{file_stem}_annotated.png")
        draw_debug_detections(winning_img, best_result["detections"], debug_annotated_path)

    # 9. Format structured records
    records = DataFormatter.to_structured_records(best_result)
    return records


def run_pipeline(
    input_path: str,
    output_dir: str,
    confidence_thresh: float = 0.50,
    roi_str: Optional[str] = None,
    debug: bool = False,
    language: str = "en",
    use_gpu: bool = False
) -> None:
    """
    Main orchestration loop for processing batch or single images.
    """
    # Build configuration
    config = PipelineConfig()
    config.input_dir = input_path
    config.output_dir = output_dir
    config.debug_mode = debug
    config.ocr.confidence_threshold = confidence_thresh
    config.ocr.language = language
    config.ocr.use_gpu = use_gpu

    setup_logging(debug)

    roi_coords = ImageLoader.parse_roi_string(roi_str) if roi_str else None

    # Discover images
    try:
        image_files = ImageLoader.get_image_files(input_path, config.supported_extensions)
    except Exception as e:
        print(f"[ERROR] Could not find input images: {e}")
        return

    if not image_files:
        print(f"[WARNING] No supported image files found at: {input_path}")
        return

    print(f"==================================================")
    print(f"  EdgeSight Industrial OCR Pipeline (Stage 1)")
    print(f"==================================================")
    print(f"Input source      : {input_path} ({len(image_files)} image(s))")
    print(f"Output directory  : {os.path.abspath(output_dir)}")
    print(f"Confidence cutoff : {confidence_thresh:.2f}")
    print(f"Language          : {language}")
    print(f"Debug mode        : {'Enabled' if debug else 'Disabled'}")
    if roi_coords:
        print(f"ROI               : {roi_coords}")
    print(f"==================================================")

    # Initialize modules
    ocr_engine = PaddleOCREngine(config.ocr)
    preprocessor = PreprocessingPipeline(config.preprocessing)

    all_records: List[Dict[str, Any]] = []
    start_time = time.time()

    # Process each image independently
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] " + "-" * 40)
        try:
            records = process_single_image(
                img_path,
                config=config,
                ocr_engine=ocr_engine,
                preprocessor=preprocessor,
                roi_coords=roi_coords
            )
            all_records.extend(records)

            # Export per-image CSV and Excel if multiple records
            if records:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                per_image_csv = os.path.join(config.csv_output_dir, f"{stem}.csv")
                per_image_xlsx = os.path.join(config.excel_output_dir, f"{stem}.xlsx")
                DataExporter.export_csv(records, per_image_csv)
                DataExporter.export_excel(records, per_image_xlsx)

        except Exception as e:
            print(f"[ERROR] Unexpected error while processing {img_path}: {e}")
            logging.exception(e)
            continue

    # Master combined export
    print("\nExporting results...")
    master_csv_path = os.path.join(config.output_dir, "results.csv")
    master_xlsx_path = os.path.join(config.output_dir, "results.xlsx")

    DataExporter.export_csv(all_records, master_csv_path)
    DataExporter.export_excel(all_records, master_xlsx_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Done.")
    print(f"Total images processed : {len(image_files)}")
    print(f"Total records saved    : {len(all_records)}")
    print(f"Elapsed time           : {elapsed:.2f}s")
    print(f"Master CSV             : {os.path.abspath(master_csv_path)}")
    print(f"Master Excel           : {os.path.abspath(master_xlsx_path)}")
    if debug:
        print(f"Processed images       : {os.path.abspath(config.processed_images_dir)}")
        print(f"Debug overlays         : {os.path.abspath(config.debug_output_dir)}")
    print("=" * 50)


def main():
    """CLI Parser and main execution entry."""
    parser = argparse.ArgumentParser(
        description="Stage 1 Industrial CCTV/Camera OCR Extraction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="input_images",
        help="Path to a single image file or directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Path to output directory for CSV, Excel, and debug files"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode to save preprocessed variants and visual overlays"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.50,
        help="Minimum OCR confidence threshold (0.0 to 1.0)"
    )
    parser.add_argument(
        "--roi", "-r",
        type=str,
        default=None,
        help="Optional Region of Interest in 'x,y,w,h' format (e.g. '100,50,400,300')"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="en",
        help="PaddleOCR language model (e.g. 'en', 'ch', 'german', 'french')"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU execution for PaddleOCR"
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        confidence_thresh=args.confidence,
        roi_str=args.roi,
        debug=args.debug,
        language=args.lang,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    main()
