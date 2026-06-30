---
title: BioSignal
emoji: 🚀
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# BioSignal — Epidemic Early Warning Intelligence Platform

**Live Demo:** [huggingface.co/spaces/Nithilasiva5125/BioSignal](https://huggingface.co/spaces/Nithilasiva5125/BioSignal)

BioSignal is a research-oriented epidemic early-warning intelligence platform that combines live WHO outbreak reports, Google Trends disease search patterns, NLP, and explainable machine learning to generate model-based public health risk signals — without relying on outcome data such as deaths or case counts.

## What It Does

- Monitors live WHO RSS outbreak reports and Google Trends search volume for 18 epidemic diseases
- Scores each report 0-100 using an XGBoost early-warning model trained on 1,093 historical WHO outbreak records
- Detects **true early warnings**: disease search spikes occurring with no WHO report yet published
- Explains every score with SHAP global and local feature importance
- Generates counterfactual what-if analysis showing what would reduce a given risk score

## Why It's Different

| What others do | What BioSignal does |
|---|---|
| Show confirmed case counts after outbreak | Detect search spikes before confirmation |
| React to WHO reports already published | Flag signals WHO has not reported yet |
| Black-box ML | SHAP + counterfactual explanations |
| One model, accuracy only | Two models — leakage demo vs. honest model |

## Model Integrity: The Leakage Demo

A key part of this project demonstrates **target leakage**. A diagnostic model trained with deaths/cases as features achieves a perfect F1=1.0 — which is a red flag, not a win, since it can't make predictions *before* deaths and cases are known. The deployed model removes all outcome features and achieves a realistic, honest **F1 = 0.889, ROC-AUC = 0.951**.

## Pipeline