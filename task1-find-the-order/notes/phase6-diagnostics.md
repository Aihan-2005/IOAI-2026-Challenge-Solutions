# Phase 6 — Ordering Diagnostics

## Current Benchmark

Official deterministic baseline:

68.95

Whisper-small + heuristic scorer + beam search:

73.45

Improvement:

+4.50 points

## Why Diagnose Before Adding Another Model?

The current pipeline has two separate sources of error:

1. transition scoring
2. global search

A stronger model is useful only if transition scoring is the actual
bottleneck.

## Local Metric

For every true conversation step:

```text
current chunk
    ->
all remaining candidate chunks