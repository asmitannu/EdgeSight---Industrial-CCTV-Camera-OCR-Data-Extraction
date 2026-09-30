# EdgeSight - Industrial CCTV & Camera OCR Data Extraction (Stage 1)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![OpenCV](https://img.shields.io/badge/vision-OpenCV-green.svg)](https://opencv.org/)

An industrial-grade optical character recognition (OCR) data extraction pipeline designed for **digital photographs captured from CCTV cameras, handheld inspection cameras, and equipment monitoring systems**.

Extracts model numbers, serial tags, electrical ratings, timestamps, and structured label tables with high accuracy under challenging industrial conditions (uneven lighting, tilted camera angles, shadows, low resolution, vibrations).

---

## Stage 1 Pipeline Architecture

```text
[ Digital Image Input ] (JPG, PNG, BMP, TIFF)
        │
        ▼
[ 1. ImageLoader ]
        ├── Safe Unicode / Windows path loading
        ├── Dimension and integrity validation
        └── Optional Region-of-Interest (ROI) cropping (x, y, w, h)
        │
        ▼
[ 2. Preprocessing & Enhancement Engine ]
        ├── Generates multiple intelligent image variants:
        │     ├── 1. Original (Base normalized image)
        │     ├── 2. Enhanced CLAHE (Local contrast in LAB color space + Unsharp Mask)
        │     ├── 3. Adaptive Threshold (Bilateral denoising + Adaptive Gaussian + Morphology)
        │     ├── 4. Otsu Threshold (Bimodal global thresholding)
        │     ├── 5. Upscaled & Sharpened (2x Bicubic interpolation + high-pass filter)
        │     ├── 6. Deskewed (Hough lines / minAreaRect angle leveling)
        │     └── 7. Perspective Corrected (4-point label contour warp)
        │
        ▼
[ 3. PaddleOCR Engine ]
        ├── Multi-variant text detection and recognition
        ├── Orientation / angle classifier
        └── Confidence score extraction per text line
        │
        ▼
[ 4. Multi-Variant Evaluator & Postprocessor ]
        ├── Composite Scoring: Confidence + Alphanumeric Density + Word Count
        ├── Winning variant selection
        └── Industrial text cleanup (preserves serial numbers, hyphens, ratings, IDs)
        │
        ▼
[ 5. Structured Data Exporter ]
        ├── results.csv (Standard RFC-4180 CSV for Siemens Insights Hub manual upload)
        ├── results.xlsx (Formatted Excel workbook with styled headers & auto-fit columns)
        └── /output/debug/ (Visual bounding-box overlay verification images)
```

---

## Key Features & Industrial Optimizations

1. **LAB-Space CLAHE Enhancement**:
   Industrial environments often suffer from harsh shadows or glare. CLAHE is applied exclusively to the Lightness (L) channel of the LAB color space, maximizing text contrast without introducing color artifacts.

2. **Bilateral Edge-Preserving Denoising**:
   Unlike standard Gaussian blur which softens character edges, bilateral filtering suppresses sensor noise from CCTV cameras while keeping sharp text boundaries intact.

3. **4-Point Perspective & Angle Deskewing**:
   CCTV cameras mounted at oblique angles capture distorted labels. The system detects quadrilateral tag contours and performs a perspective warp to generate a flat top-down view.

4. **Multi-Variant OCR Strategy**:
   Instead of guessing a single preprocessing configuration, the system generates targeted variants, evaluates OCR results across all of them, and automatically selects the optimal extraction.

5. **Industrial Serial & Model ID Preservation**:
   Postprocessing normalizes whitespace and removes control characters while strictly safeguarding hyphens, slashes, decimal points, and alphanumeric patterns (e.g. `1LE1001-1DB43-4AA4`, `SN-98420194/01`, `400V 50Hz`).

---

## Directory Structure

```text
EdgeSight/
│
├── main.py                          # CLI entry point with batch processing
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation & guides
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Global configurations, thresholds & flags
│
├── input_images/                    # Input folder for photographs and CCTV frames
│
├── output/                          # Generated outputs
│   ├── results.csv                  # Master combined CSV
│   ├── results.xlsx                 # Master combined formatted Excel workbook
│   ├── csv/                         # Per-image CSV files
│   ├── excel/                       # Per-image Excel workbooks
│   ├── processed_images/            # Saved preprocessing variants (when --debug)
│   └── debug/                       # Visual bounding box overlays (when --debug)
│
├── src/
│   ├── __init__.py
│   ├── image_loader.py              # Safe path loading, validation, ROI cropping
│   ├── image_enhancement.py         # CLAHE, denoising, sharpening, upscaling, morphology
│   ├── perspective_correction.py    # Deskewing & 4-point perspective warp
│   ├── preprocessing.py             # PreprocessingPipeline orchestrator
│   ├── ocr_engine.py                # PaddleOCR engine wrapper
│   ├── postprocessing.py            # Text normalization & ID preservation
│   ├── data_formatter.py            # Variant scoring & structured record builder
│   └── exporter.py                  # CSV and Excel export with pandas & openpyxl
│
└── tests/
    ├── __init__.py
    ├── test_image_loader.py         # Unit tests for image loader & ROI
    ├── test_preprocessing.py        # Unit tests for filters & enhancement
    ├── test_perspective.py          # Unit tests for deskewing & warp
    ├── test_postprocessing.py       # Unit tests for text cleanup
    ├── test_formatter_exporter.py   # Unit tests for scoring & exports
    └── generate_sample_images.py    # Synthetic test image generator
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12** (64-bit)
- Windows / Linux / macOS

### 2. Activate Virtual Environment
```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start & Usage

### 1. Generate Synthetic Sample Images (Optional)
To test the pipeline immediately with realistic sample equipment plates:
```bash
python tests/generate_sample_images.py
```
This generates:
- `input_images/equipment_motor_01.png` (Siemens Simotics motor nameplate)
- `input_images/equipment_tilted_tag_02.png` (Tilted equipment tag with background noise)
- `input_images/equipment_cctv_shadow_03.png` (Low-contrast CCTV shadow frame)

---

### 2. Process All Images in Input Directory
```bash
python main.py --input input_images/
```

Console Output:
```text
==================================================
  EdgeSight Industrial OCR Pipeline (Stage 1)
==================================================
Input source      : input_images/ (3 image(s))
Output directory  : C:\...\EdgeSight\output
Confidence cutoff : 0.50
Language          : en
Debug mode        : Disabled
==================================================

[1/3] ----------------------------------------
Processing image: equipment_motor_01.png
Generating preprocessing variants...
Running OCR...
OCR completed. (Selected variant: 'original', Detections: 7)
  -> 'SIEMENS SIMOTICS 3~ MOT' (confidence: 0.98)
  -> 'MODEL: 1LE1001-1DB43-4AA4' (confidence: 0.97)
  -> 'SERIAL NO: SN-98420194-01' (confidence: 0.96)
  ...

Exporting results...
==================================================
Done.
Total images processed : 3
Total records saved    : 19
Master CSV             : C:\...\EdgeSight\output\results.csv
Master Excel           : C:\...\EdgeSight\output\results.xlsx
==================================================
```

---

### 3. Process a Single Image with Debug Mode
Debug mode saves all intermediate preprocessing variants to `output/processed_images/` and visual bounding-box overlays to `output/debug/`.
```bash
python main.py --input input_images/equipment_motor_01.png --debug
```

---

### 4. Process a Specific Region of Interest (ROI)
If text is located in a specific area of a high-resolution CCTV frame:
```bash
# Format: --roi x,y,width,height
python main.py --input input_images/equipment_motor_01.png --roi 40,100,700,300
```

---

### 5. Custom Confidence Threshold
Filter out lower-confidence OCR detections:
```bash
python main.py --input input_images/ --confidence 0.75
```

---

### 6. Command Line Reference

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `input_images` | Path to image file or directory containing images |
| `--output` | `-o` | `output` | Output directory for CSV, Excel, and debug files |
| `--debug` | `-d` | `False` | Save preprocessing variants & visual bounding box overlays |
| `--confidence` | `-c` | `0.50` | Minimum confidence threshold (0.0 to 1.0) |
| `--roi` | `-r` | `None` | Region of Interest in `x,y,w,h` format |
| `--lang` | `-l` | `en` | OCR language model (`en`, `ch`, `german`, `french`, etc.) |
| `--no-gpu` | | `False` | Force CPU execution for PaddleOCR |

---

## Structured Output Format

The output schema is designed for seamless import into spreadsheets, analytics engines, and manual upload into **Siemens Insights Hub**:

### Output Schema: `results.csv` and `results.xlsx`

| Image Name | Detected Text | Confidence | Bounding Box | Processing Variant | Timestamp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `equipment_motor_01.png` | `SIEMENS SIMOTICS 3~ MOT` | `0.9842` | `[[40,45],[410,45],[410,65],[40,65]]` | `original` | `2026-08-25T11:45:00Z` |
| `equipment_motor_01.png` | `MODEL: 1LE1001-1DB43-4AA4` | `0.9715` | `[[40,120],[480,120],[480,142],[40,142]]` | `original` | `2026-08-25T11:45:00Z` |
| `equipment_motor_01.png` | `SERIAL NO: SN-98420194-01` | `0.9630` | `[[40,170],[490,170],[490,192],[40,192]]` | `original` | `2026-08-25T11:45:00Z` |
| `equipment_cctv_shadow_03.png`| `VOLTAGE: 400V / 690V 50Hz` | `0.9420` | `[[40,220],[520,220],[520,242],[40,242]]` | `enhanced_clahe` | `2026-08-25T11:45:02Z` |

---

## Running Unit Tests

Run the complete test suite using `pytest`:
```bash
.\.venv\Scripts\pytest -v
```

---

## Future Roadmap (Stages 2 & 3)

The project architecture is modular to allow future stages to integrate smoothly:
- **Stage 2**: Live CCTV RTSP/video stream frame-grabbing, motion-triggered capture, and local SQLite/disk buffering for intermittent network conditions.
- **Stage 3**: Automated REST API synchronization and MQTT publishing to **Siemens Insights Hub (MindSphere)**.
