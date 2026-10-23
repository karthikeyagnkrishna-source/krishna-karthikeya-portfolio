# V7 Visual History Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Execute inline to honor the user's efficient-credit preference. Steps use checkbox syntax.

**Goal:** Implement timestamp-aware past/current visual memory in V7 while retaining the complete robot history and original prediction/loss contracts.

**Architecture:** Reuse V7's image resamplers, causal robot encoders, four query-fusion blocks and all prediction heads. Five current queries read six image encodings plus all encoded robot samples. A zero-initialized residual projection preserves pretrained outputs; the foundation remains frozen by default.

**Tech Stack:** existing Python, NumPy and PyTorch environment on UGP_USER; existing foundation.ipynb definition loader.

**Spec:** docs/superpowers/specs/2026-10-10-v7-visual-history-design.md

## Global constraints

- Current plus five immediately preceding camera frames; exact relative timestamps and missing-frame masks.
- Existing 50/400 robot samples and +0.1/+0.2...+1.0 s horizons; no hand-assigned image/sample correspondence.
- Reuse raw image feature caches; old frozen context caches are incompatible with this revision.
- Preserve sources/, all previous implementations and all previous checkpoints.
- User requested coding; do not launch substantive training.
- This project mirror is not a Git checkout. Work in a new package; no worktree or commits are available.

## Review focus

1. A future image or shifted robot timestamp must be rejected.
2. Missing early images must be masked and not affect outputs.
3. Past-image perturbation must affect the model after opening the residual adapter.
4. The frozen original foundation must receive no gradients; history adapters must receive gradients.
5. Original predictions and output shapes must match exactly at the zero warm start on both history lengths.

## Task 1: timestamped input contract

- [x] Write and run failing consecutive-frame selection tests.
- [x] Create v7_visual_history/data.py: HistoryBatch, select_camera_history(camera_times, anchor, count=6), build_history(data, visual, idx, steps, device, count=6).
- [x] Verify native camera timestamps, early padding, future exclusion and missing-frame gaps.

## Task 2: reuse V7 with temporal memory

- [x] Write failing warm-start, temporal sensitivity, mask and gradient tests using actual V7 class definitions.
- [x] Create v7_visual_history/model.py: V7VisualHistory(base, frames=6), forward_sample(sample), encode_sample(sample), features(sample).
- [x] Return original V7 prediction keys and per-leg 1285-wide downstream force features.
- [x] Verify 554/2304 memory-token counts; retain original causal full sequences.

## Task 3: runtime, ready-to-run training entry point and delivery

- [x] Create runtime.py to attach history to original samples and load the existing V7 checkpoints; reuse the original trainer, loss and evaluator hooks in a separate runtime.
- [x] Create run.py with explicit --train and default read-only --check; separate output namespace and manifest/signature checks.
- [x] Run all new tests and one CPU integration check with real short/long data and saved foundation weights; run the existing specialist regression suite.
- [x] Review, fix concrete findings, document usage and limitations, and package verified source. Do not launch training.

