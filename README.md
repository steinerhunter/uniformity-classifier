# Uniformity Test Pass/Fail Classifier

A machine learning pipeline for classifying GE HealthCare scanner uniformity tests as PASS or FAIL.

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

# 5. Run the pipeline
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

## Project Structure

```
uniformity-classifier/
├── README.md                 # This file
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
│   └── evaluate.py           # Metrics and comparison
├── tests/                    # Unit tests
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_evaluate.py
├── outputs/                  # Generated results
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_advanced.png
│   └── results.json
└── REPORT.md                 # Summary report (1-2 pages)
```

## Models

### Baseline: Random Forest on Hand-Crafted Features

Extracts interpretable image quality metrics:
- Intensity statistics (mean, std, coefficient of variation)
- Spatial analysis (gradient magnitude)
- Artifact detection (local variance hotspots)
- Histogram analysis (entropy, percentile range)

### Advanced: GPT-4o Vision

Uses OpenAI's GPT-4o to analyze images with natural language reasoning.
Provides interpretable explanations for each classification.

**Note:** Requires `OPENAI_API_KEY` environment variable. Results are cached for offline reproducibility.

## Environment Variables

```bash
# Required for advanced model
export OPENAI_API_KEY=your_api_key_here
```

## Running Tests

```bash
pytest tests/ -v
```

## Outputs

After running `python main.py`, you'll find:

- `outputs/results.json` - Full metrics for both models
- `outputs/confusion_matrix_baseline.png` - Baseline confusion matrix
- `outputs/confusion_matrix_advanced.png` - Advanced model confusion matrix
- Console output with side-by-side comparison

## Offline Mode

The advanced model caches all API responses in `cache/gpt4o_responses.json`.
Once you've run the pipeline, it can be reproduced without API access.

## Requirements

- Python 3.9+
- No GPU required (runs on CPU)
- ~4GB RAM

## License

This project was created as a take-home assignment for GE HealthCare.
