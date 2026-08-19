# Task 1 — Baseline Investigation

## Problem

The goal is to reconstruct the chronological order of shuffled audio turns
from a two-speaker English dialogue.

Each dialogue contains between 7 and 20 WAV chunks.

The filename index is only the shuffled index and does not reveal the
chronological position.

## Prefix Information

`prefix.json` gives the indexes of the true first and second chunks.

Example:

```text
prefix = [1, 2]