# Pill Identification Matching Logic

This document explains the matching logic behind `POST /pill_identify/`.

## Overview

The endpoint does two things:

1. **Predict attributes from an image**
   - **Shape** via YOLO classifier
   - **Imprint** via TrOCR (OCR)
   - **Colors** via a multi-label classifier

2. **Match predictions to the pill database**
   - Query the `pills` table (seeded from `smart_drug_assistant/db/pill_color_imprint_dataset.csv`)
   - Return a ranked list of candidate pills (`matches`) rather than a single hard decision

This design is intentional: pill OCR and color prediction are not perfectly reliable, so the API prioritizes **high recall + explainable ranking**.

## Database Schema

The matcher queries the `pills` table:

- `image_path`
- `image_filename`
- `pill_name`
- `shape`
- `color` (string, may contain multiple colors like `"BLUE, WHITE"`)
- `imprint` (string, often semicolon-separated tokens like `WATSON;384`)

## Step 1 — Model Predictions

### Shape

- Predicted using YOLO.
- Returned as:
  - `shape.class`
  - `shape.confidence`

### Color

- Predicted using a sigmoid multi-label head.
- The endpoint returns:
  - `colors`: thresholded labels
  - `color_scores`: top scores (debug/visibility)

### Imprint (OCR)

TrOCR sometimes produces overly long/noisy strings (hallucinated extra digits, merged tokens, etc.).

To mitigate this, the OCR pipeline applies:

- **Constrained decoding** (shorter, less repetitive sequences)
- **Canonicalization** to move output toward the DB’s imprint format (semicolon-separated tokens)

The goal is not to “invent” missing characters, but to **reduce spurious characters** so matching is stable.

## Step 2 — Canonical OCR Imprint Formatting

The matcher normalizes OCR text to a DB-like format:

- Uppercase
- Keep only `A-Z`, `0-9`, and `;`
- Collapse repeated separators
- Extract alpha/digit chunks even when OCR merges them

Heuristics:

- Manufacturer-like tokens (longer alphabetic) often pair with a **3-digit** code (e.g. `WATSON;384`).
- 2-letter tokens often pair with **4-digit** codes (e.g. `TV;1003`).
- 1-letter tokens can be 3- or 4-digit; logic prefers a 4-digit code if present.

## Step 3 — Query Strategy (Progressive Fallback)

The endpoint uses multiple query passes to avoid false negatives:

1. **If an imprint exists**: query by `shape + imprint` first **without color filtering**.
   - Rationale: color prediction is frequently noisier than OCR.

2. If no hits: try adding color filtering.

3. If still no hits: relax by dropping imprint, then dropping colors.

4. If still no hits and shape confidence is low: drop shape.

This ensures the API returns candidates even when one attribute is wrong.

## Step 4 — Color Filtering

When multiple colors are chosen (e.g. `BLUE` vs `TURQUOISE` close scores), color filtering uses **OR** semantics:

- `color ILIKE '%BLUE%' OR color ILIKE '%TURQUOISE%'`

This prevents incorrectly requiring the DB color string to contain all predicted colors.

## Step 5 — Imprint Fuzzy Matching

The DB stores imprints as token-like strings (often semicolon-separated). Matching is performed with:

- **Tokenization** of both predicted imprint and DB imprint
- A lightweight **edit-distance similarity** scoring
- A small “confusable character” mapping for OCR ambiguity in the regex layer (e.g. `O/0`, `I/1/L`, `S/5`)

The API returns a `match` object per candidate containing:

- `imprint_similarity`
- `best_predicted_token`
- `best_imprint_token`
- `shape_match`
- `color_match_count`

## Why This Approach

- **User-facing robustness**: returning a ranked list is more reliable than returning a single hard label.
- **Explainability**: the response contains enough signals to explain why a pill was suggested.
- **Practicality for demos**: small rule-based corrections often yield large improvements without expensive retraining.

## Known Limitations

- If OCR drops digits entirely (e.g. `3566` → `35`), canonicalization cannot recover them.
- Some codes are highly ambiguous (e.g. many pills share `300`). In these cases ranking should consider multiple tokens.
- Color classification can be confused by lighting, glare, or similar hues (e.g. `PINK` vs `RED`).


