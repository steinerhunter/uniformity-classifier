# Uniformity Test Pass/Fail Classifier

[![View Full Report](https://img.shields.io/badge/📊_View_Full_Report-Documentation-blue?style=for-the-badge)](https://steinerhunter.github.io/uniformity-classifier/)

A machine learning pipeline for classifying GE HealthCare scanner uniformity tests as PASS or FAIL.

> **Note:** The test images are not included in this repository. You will need to add your own DICOM images to the `data/` directory (see Data Setup below).

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/steinerhunter/uniformity-classifier.git
cd uniformity-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your data (see Data Setup below)

# 5. Set OpenAI API key (for advanced model)
export OPENAI_API_KEY=your_api_key_here

# 6. Run the pipeline
python main.py
```

## Data Setup

Place your data in the `data/` directory with this structure:

```
data/
├── PASS/
│   ├── image_001.dcm (or .png/.jpg)
│   ├── image_002.dcm
│   └── ...
└── FAIL/
    ├── image_003.dcm
    ├── image_004.dcm
    └── ...
```

The pipeline supports both DICOM (`.dcm`) and standard image formats (`.png`, `.jpg`).

## Command Line Options

```bash
python main.py --help

Usage: python main.py [OPTIONS]

Options:
  --data-dir PATH      Directory containing PASS/ and FAIL/ subdirs (default: ./data)
  --output-dir PATH    Directory for output files (default: ./outputs)
  --cache-dir PATH     Directory for API response cache (default: ./cache)
  --test-size FLOAT    Fraction of data for testing (default: 0.2)
  --random-state INT   Random seed for reproducibility (default: 42)
  --no-advanced        Skip the GPT-4o model (baseline only)
  --verbose, -v        Enable debug logging

Examples:
  python main.py                              # Run with defaults
  python main.py --data-dir ./custom_data     # Custom data directory
  python main.py --test-size 0.3              # 30% test split
  python main.py --no-advanced                # Skip GPT-4o (faster)
  python main.py --verbose                    # Debug output
```

## Project Structure

```
uniformity-classifier/
├── README.md                 # This file
├── REPORT.md                 # Summary report (1-2 pages)
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point - runs full pipeline
├── data/                     # Your dataset (not committed)
│   ├── PASS/
│   └── FAIL/
├── cache/                    # Cached GPT-4o responses (for offline mode)
├── src/
│   ├── data_loader.py        # Load and split dataset
│   ├── features.py           # Feature extraction for baseline
│   ├── baseline.py           # Random Forest classifier
│   ├── advanced.py           # GPT-4o vision classifier
│   └── evaluate.py           # Metrics, comparison, visualization
├── tests/                    # Unit tests (39 tests)
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_evaluate.py
└── outputs/                  # Generated results
    ├── sample_images.png
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_advanced.png
    ├── per_image_results.csv
    └── results.json
```

## Models

### Baseline: Random Forest on Hand-Crafted Features

Extracts 8 interpretable image quality metrics:

| Feature | Description |
|---------|-------------|
| `mean_intensity` | Average pixel brightness |
| `std_intensity` | Standard deviation of pixel values |
| `coef_of_variation` | Normalized variability (std/mean) |
| `gradient_magnitude` | Edge/transition detection via Sobel |
| `max_local_variance` | Artifact hotspot detection |
| `histogram_entropy` | Pixel distribution spread |
| `percentile_range` | Robust spread measure (p95-p5) |
| `center_vs_edge_ratio` | Vignetting detection |

### Advanced: GPT-4o Vision

Uses OpenAI's multimodal LLM to analyze images with natural language reasoning.
Provides interpretable explanations for each classification.

## Outputs

After running `python main.py`, you'll find:

| File | Description |
|------|-------------|
| `sample_images.png` | Visualization of PASS vs FAIL examples |
| `confusion_matrix_baseline.png` | Baseline model confusion matrix |
| `confusion_matrix_advanced.png` | GPT-4o model confusion matrix |
| `per_image_results.csv` | Per-image breakdown with predictions |
| `results.json` | Full metrics, feature importances, comparison |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Offline Mode

The advanced model caches all API responses in `cache/gpt4o_responses.json`.
Once you've run the pipeline, it can be reproduced without API access:

```bash
# First run - makes API calls and caches
python main.py

# Subsequent runs - uses cache
python main.py  # No API calls needed
```

## Requirements

- Python 3.9+
- No GPU required (runs on CPU)
- ~100MB RAM

## License

This project was created as a take-home assignment for GE HealthCare.
