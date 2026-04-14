# BCI_RSVP_AUTH

EEG-based RSVP authentication pipeline for analyzing P300-like responses and classifying target vs non-target stimuli.

## Overview

This project is a small Brain-Computer Interface (BCI) authentication prototype inspired by RSVP-based EEG login systems.  
The core idea is simple: a user mentally attends to chosen password images inside a rapid stream of visual stimuli, and the EEG response is used to distinguish **target** images from **non-target** images.

This repository focuses on the **offline analysis pipeline**, including:

- EEG loading and preprocessing
- event extraction and epoching
- target/non-target ERP analysis
- simple ocular artifact rejection
- feature extraction in the P300 time window
- binary classification with cross-validation

It is **not** a full deployed authentication system yet. It does not include a real-time interface, password enrollment logic, online stopping criteria, or a production authentication backend.

---

## Project Goal

The goal of this project is to test whether RSVP-evoked EEG responses contain enough information to separate attended password-related stimuli from irrelevant stimuli.

In this setup:

- **Target stimuli** = the user’s selected password images
- **Non-target stimuli** = all other images shown in the RSVP stream

The project studies an **ERP / P300-style oddball response**, where target items tend to evoke stronger centro-parietal positivity than non-target items.

---

## Repository Contents

Typical files used in this project:

- `BCI_EEG_lab1.ipynb` — main notebook for step-by-step analysis
- `test_bci_clf.py` — classifier test script
- `RSVP_login_VPeal.vhdr` / `.vmrk` / `.eeg` — BrainVision EEG recording
- `RSVP_login_dry_VPeal.vhdr` / `.vmrk` / `.eeg` — dry-electrode version
- `RSVP_login_VPeal_100Hz_eeg.fif` — preprocessed FIF EEG file
- `AHigh-SecurityEEG-BasedLoginSystemwithRSVPStimuliandDryElectrodes-1.pdf` — reference paper
- `Part 1.docx` — project/lab write-up

---

## Data Format

This project uses EEG recordings stored in **BrainVision format**:

- `.vhdr` — header file
- `.vmrk` — marker file
- `.eeg` — signal data

These files must stay together in the same folder.

A preprocessed `.fif` file is also included for MNE-based workflows.

---

## Processing Pipeline

### 1. Load EEG data
The raw EEG is loaded with **MNE-Python** from a BrainVision file.

### 2. Channel setup
- `EOGv1` is renamed to `Fp1`
- a standard `10-20` montage is assigned

### 3. Preprocessing
The current pipeline applies:

- low-pass filtering at **40 Hz**
- resampling to **100 Hz**

This keeps the signal focused on ERP-relevant low frequencies and reduces data size.

### 4. Event extraction and epoching
Events are extracted from annotations and split into:

- `T1`, `T2`, `T3` → combined into **Target**
- `NT1` ... `NT22` → combined into **Non-Target**

Epochs are created using:

- `tmin = -0.2 s`
- `tmax = 1.0 s`
- baseline correction from pre-stimulus interval

### 5. Artifact rejection
A simple ocular artifact rejection rule is used:

- VEOG-related peak-to-peak amplitude is checked
- epochs above a threshold are removed

This is a lightweight approach, but it is intentionally simple and not state-of-the-art.

### 6. Feature extraction
Features are extracted from a centro-parietal P300-relevant region using channels such as:

- `Cz`
- `Pz`
- `CP1`
- `CP2`
- `P3`
- `P4`
- `CPz` (if available)

The main feature representation uses:

- **200–600 ms** post-stimulus window
- flattened **channel × time** representation
- amplitude converted to **µV**

An alternative feature set based on time-window averages is also tested.

### 7. Classification
The repository evaluates:

- **Shrinkage LDA**
- **Balanced Logistic Regression**

Evaluation is done using **5-fold stratified cross-validation** with:

- **ROC AUC**
- **Balanced Accuracy**

These metrics are preferred because the dataset is imbalanced.

---

## Why not plain accuracy?

In RSVP data, non-targets are much more frequent than targets.  
A classifier that always predicts “non-target” could still get a deceptively high accuracy.

That is why this project uses:

- **Balanced Accuracy**
- **ROC AUC**

instead of relying only on standard accuracy.

---

## Current Findings

The analysis shows that:

- target trials produce stronger positive deflections in the P300 range
- the centro-parietal ROI captures this effect well
- the target-minus-non-target difference is visible both visually and quantitatively
- single-trial classification achieves reasonable above-chance performance
- averaging more repetitions strongly improves separability

Example reported results from this project:

- single-trial **ROC AUC ≈ 0.79**
- single-trial **Balanced Accuracy ≈ 0.71**
- aggregated performance improves substantially with more repetitions

---

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install mne numpy scikit-learn jupyter
