# Uniformity Test Pass/Fail Classification - Summary Report

## 1. Pipeline Overview

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Loading   │────▶│   Train/Test     │────▶│   Evaluation    │
│  (DICOM/PNG)    │     │     Split        │     │   & Comparison  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│   Baseline Model        │       │   Advanced Model        │
│   ─────────────────     │       │   ──────────────────    │
│   1. Feature extraction │       │   1. Image → GPT-4o     │
│   2. Random Forest      │       │   2. JSON response      │
│   3. Prediction         │       │   3. Cached for offline │
└─────────────────────────┘       └─────────────────────────┘
```

### Data Flow

1. **Input:** DICOM or PNG images organized in `PASS/` and `FAIL/` folders
2. **Split:** 80% training, 20% testing (stratified)
3. **Baseline:** Extract 8 hand-crafted features → Random Forest classifier
4. **Advanced:** Send images to GPT-4o Vision → Get classification + reasoning
5. **Output:** Confusion matrices, metrics comparison, JSON results

---

## 2. Assumptions

- **Data format:** Images are already cropped to the phantom region
- **Class balance:** Dataset has reasonable balance; class weights used to handle minor imbalance
- **Image quality:** Input images are readable and not corrupted
- **API reliability:** GPT-4o responses are deterministic with temperature=0
- **Ground truth:** Labels are accurate and consistent

---

## 3. Results

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total samples | TBD |
| PASS samples | TBD |
| FAIL samples | TBD |
| Test set size | TBD |

### Baseline Model (Random Forest)

**Confusion Matrix:**

![Baseline Confusion Matrix](outputs/confusion_matrix_baseline.png)

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |

**Top 5 Most Important Features:**

1. TBD
2. TBD
3. TBD
4. TBD
5. TBD

### Advanced Model (GPT-4o Vision)

**Confusion Matrix:**

![Advanced Confusion Matrix](outputs/confusion_matrix_advanced.png)

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |

**Sample Reasoning:**

> *Example 1:* "TBD - GPT-4o's explanation for a classification"

> *Example 2:* "TBD - Another example"

---

## 4. Model Comparison

| Metric | Baseline | GPT-4o | Winner |
|--------|----------|--------|--------|
| Accuracy | TBD | TBD | TBD |
| Precision | TBD | TBD | TBD |
| Recall | TBD | TBD | TBD |
| F1 Score | TBD | TBD | TBD |

### Analysis

TBD - Analysis of which model performed better and why.

---

## 5. LLM Evaluation Details

### Prompting Strategy

The GPT-4o model was prompted as a "medical imaging QA specialist" with explicit instructions to:
1. Look for brightness variations and gradients
2. Identify artifacts (rings, bands, spots)
3. Detect signal dropouts
4. Provide structured JSON output with classification, confidence, and reasoning

### Strengths

- Provides human-readable explanations
- Can identify novel failure modes not captured by hand-crafted features
- Zero training required

### Limitations

- API cost and latency
- Requires internet connection (mitigated by caching)
- Less consistent than trained classifier

---

## 6. What I Would Do Differently

### With More Data

- **Fine-tune a vision model:** Use a pre-trained CNN (ResNet, EfficientNet) and fine-tune on this specific task
- **Data augmentation:** Rotate, flip, add noise to increase training set size
- **Cross-validation:** Use k-fold cross-validation for more robust evaluation

### With More Compute

- **Ensemble methods:** Combine multiple classifiers (voting ensemble)
- **Hyperparameter optimization:** Grid search or Bayesian optimization for Random Forest
- **Vision transformer:** Try ViT or CLIP embeddings + classifier head

### With More Time

- **Diffusion model approach:** Use a diffusion model to learn the "uniform" distribution, flag deviations
- **Segmentation:** Automatically segment the phantom region before classification
- **Explainability:** Add Grad-CAM or SHAP visualizations for the baseline model
- **Error analysis:** Deep dive into misclassified samples to understand failure modes
- **Domain expert consultation:** Work with radiologists/physicists to validate feature engineering

---

## Appendix: Reproducibility

```bash
# Clone and setup
git clone https://github.com/steinerhunter/uniformity-classifier.git
cd uniformity-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add data
# Place images in data/PASS/ and data/FAIL/

# Set API key (for advanced model)
export OPENAI_API_KEY=your_key_here

# Run
python main.py

# Results appear in outputs/
```
