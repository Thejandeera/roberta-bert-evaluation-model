# Emotion Detection: RoBERTa vs. BERT

A suite for benchmarking emotion detection models in business call
environments.

## 1. Setup Virtual Environment

Run these commands in your project folder.

### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Dependencies

Install the required libraries:

```bash
pip install pandas transformers openpyxl plotly
```

## 3. Execution

### A. Run Benchmark

This downloads the models, processes `input.csv`, and generates
`comparison.html` and `evaluation_results.xlsx`.

```bash
python benchmark.py
```

### B. Interactive Testing

Use this for real-time, character-by-character testing of custom
strings.

```bash
python main.py
```

## Data Files

- **input.csv**: Place custom inputs here (required columns: `text`,
  `actual_emotion`).
- **evaluation_results.xlsx**: Auto-generated report with color-coded
  results:
  - 🟨 Yellow = Match
  - 🟥 Red = Mismatch
- **comparison.html**: Visual distribution chart of model predictions.
