# Phase 3 — Local Solution Architecture

## Objective

Build the complete inference skeleton before introducing
GPU-dependent models.

## Architecture

```text
IOAI Dataset
    |
    v
dataset_io.py
    |
    v
Dialogue
    |
    v
RankingPredictor
    |
    v
pipeline.py
    |
    v
submission.py
    |
    v
answers.json
