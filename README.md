# Audio Deepfake Detection (SNAC + SafeEar)

## Overview
This project explores deepfake audio detection using:
- SNAC (Semantic Neural Audio Codec)
- SafeEar architecture

The goal is to study the tradeoff between:
- Audio compression (SNAC)
- Detection performance (SafeEar)

## What I did
- Integrated SNAC representations into SafeEar
- Replaced original feature pipeline
- Evaluated performance using STOI / PESQ

## Results
- SNAC preserved content better than baseline
- Detection performance dropped slightly due to feature mismatch

## Project Structure
- safe_snac/ → main implementation
- snac_integration/ → integration code
- train_snac_safeear.py → training script

## How to Run
1. Activate virtual environment

   conda activate safe_snac_env
   
2. Add dataset (not included)
3. Run:
   python train_snac_safeear.py

## Note
Datasets and model checkpoints are not included due to size.
