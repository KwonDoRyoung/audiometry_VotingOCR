# ocr_audiometry

A **zero-shot OCR pipeline** for extracting patient information and PTA/SRT/SDT measurements from standardized **audiometry report scans (JPG)**.
The system runs without any training data — it relies solely on the pretrained Tesseract LSTM model combined with deterministic OpenCV-based preprocessing.

---

## 1. Extraction Targets

| Field | Content |
|---|---|
| Patient info | Age, examination dates (2) |
| **PTA** (Pure-Tone Audiometry) | 12 frequencies × 2 conductions (AC/BC) × 2 sides (R/L) = **48 values** |
| **SRT** (Speech Recognition Threshold) | One dB value per side (R/L) |
| **SDT** (Speech Discrimination Test) | (%, stimulus dB) pair per side (R/L) |

---

## 2. Setup

```bash
pip install -r requirements.txt
# tesseract-ocr must be installed on the host system
```

Key dependencies: `opencv-python`, `pytesseract`, `scipy`, `pandas`, `openpyxl`, `tqdm`.
See [requirements.txt](requirements.txt) for exact versions.

---

## 3. Quick Start

```bash
python main.py \
  --info-path <metadata xlsx> \
  --data-root-path <JPG directory> \
  --save-path <output xlsx> \
  --is-verification \
  --choice-string-operator dilate3x3 dilate2x2 open2x2 close3x3
```

### Main Options

| Flag | Role |
|---|---|
| `--is-verification` | Enable multi-candidate verification engine |
| `--choice-string-operator` | Morphology operators used by Path A (0–4 of them) |
| `--off-char-checker` | Disable per-character OCR (Path B) |
| `--off-remove-symbol` | Disable unit-symbol removal heuristic |
| `--eval` | Process only rows where `Checked == 1` |
| `--debug` | Sample 0.1% of the data |

Parallelism: `ProcessPoolExecutor` with `max_workers = min(32, cpu_count - 4)`.

---

## 4. Pipeline

```
Input JPG (BGR → Gray)
    ├──► P.Info  (ROI: y[0,160] × x[350,800])
    ├──► PTA     (ROI: y[550,690] × x[0,550])
    ├──► SRT     (ROI: y[710,870] × x[0,350])
    └──► SDT     (ROI: y[710,870] × x[350,570])

Per module:
   Fixed-ROI crop → projection-based trim → ×4 upscale → Otsu binarize
   → Canny + 1D peak detection → table grid recovery (fail-safe assertion)
   → per-cell morphology + Otsu
   → Verification Engine
        ├─ Path A: whole-string OCR × N morphology variants
        └─ Path B: per-character OCR with positional whitelists
   → majority voting (length-consistency filter + domain priors)
   → (string, confidence ∈ [0, 100])
```

Full algorithm, parameters, and ablation matrix are documented in [paper_methodology.md](paper_methodology.md).

---

## 5. Key Contributions

1. **Unit-Symbol Removal** — resolves cells where digits and unit glyphs (`40dB`, `96%`) are printed together, using a pixel-width statistical heuristic. No training data required.
2. **Multi-Candidate Verification** — combines a morphology-variant ensemble (Path A) with per-digit OCR under positional whitelists (Path B); fused via majority voting with domain priors.
3. **Cross-Cell Context Correction** — exploits R/L and AC/BC pairing consistency to recover partial failures while rejecting any character outside the whitelist (fail-safe).
4. **Confidence ≠ raw match rate** — length-mismatch candidates remain in the denominator, acting as an implicit penalty. Reviewing only `confidence < 100%` cells minimizes human-in-the-loop cost.

---

## 6. Directory Layout

```
ocr_audiometry/
├── main.py                  # entry point (parallel OCR)
├── ocr/
│   ├── ocr_pinfo.py         # patient info module
│   ├── ocr_pta.py           # PTA module
│   ├── ocr_srt.py           # SRT module
│   ├── ocr_sdt.py           # SDT module
│   ├── vision/              # grid recovery, unit-symbol removal, etc.
│   └── utils/
└── requirements.txt
```

---

## 7. Limitations / Notes

- All heuristic thresholds are defined in the **×4 upscaled coordinate system**. Porting to another resolution requires proportional rescaling.
- The `find_peaks` height thresholds depend on page DPI — retune if input resolution changes.
- ROI coordinates are hard-coded on the assumption that the report template is fixed. If the template changes, update `crop_region` in `ocr/*.py`.
- Confidence of 100% does not guarantee correctness (all candidates may share the same error). Reviewing only `< 100%` cells is a deliberate **recall-first policy**.
