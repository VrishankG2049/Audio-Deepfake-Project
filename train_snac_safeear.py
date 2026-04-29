import torch
from snac_integration.snac_wrapper import SNACWrapper

# Use CPU first to avoid GPU issues
device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] Using device:", device)

# Create dummy audio (1 sec at 24kHz)
audio = torch.randn(1, 1, 24000)

# Initialize SNAC
snac = SNACWrapper(device=device)

# Encode
tokens = snac.encode(audio)
from snac_integration.token_processing import tokens_to_features

features = tokens_to_features(tokens)
from snac_integration.token_processing import prepare_acoustic_features

acoustic = prepare_acoustic_features(features, mode="T3_T4")

# Print info
print("\n[INFO] Number of token scales:", len(tokens))

print("Final acoustic feature shape:", acoustic.shape)

