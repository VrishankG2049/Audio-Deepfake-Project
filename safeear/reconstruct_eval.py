import torch
import torchaudio
import numpy as np

from pystoi import stoi
from pesq import pesq

from omegaconf import OmegaConf
from safeear.trainer.safeear_trainer import SafeEarTrainer
from safeear.datas.asvspoof19 import DataModule


# -----------------------------
# LOAD CONFIG (same as training)
# -----------------------------
cfg = OmegaConf.load("config/train19.yaml")

# -----------------------------
# LOAD SYSTEM (this loads SNAC correctly)
# -----------------------------
from hydra.utils import instantiate

# build models from config (same as training)
decouple_model = instantiate(cfg.decouple_model)
detect_model = instantiate(cfg.detect_model)

# now load system correctly
system = SafeEarTrainer.load_from_checkpoint(
    "Exps/ASVspoof19/checkpoints/epoch=0-val_eer=0.3976.ckpt",
    decouple_model=decouple_model,
    detect_model=detect_model,
    lr_raw_former=cfg.system.lr_raw_former,
    save_score_path=cfg.system.save_score_path
)

system = system.cuda()
system.eval()

# 🔥 THIS IS YOUR SNAC MODEL
snac = system.decouple_model
snac.eval()


# -----------------------------
# LOAD AUDIO
# -----------------------------
audio_path = "/home/zeta/Workbenches/KA/Dataset/asvspoof/LA/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac"

waveform, sr = torchaudio.load(audio_path)

if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

waveform = waveform.cuda()

# 🔥 ADD THIS LINE
waveform = waveform.unsqueeze(1) if waveform.dim() == 2 else waveform


with torch.no_grad():
    tokens = snac.encode(waveform)   # encode
    reconstructed = snac.decode(tokens)  # decode


# -----------------------------
# METRICS
# -----------------------------
original = waveform.squeeze().cpu().numpy()
reconstructed = reconstructed.squeeze().cpu().numpy()

min_len = min(len(original), len(reconstructed))
original = original[:min_len]
reconstructed = reconstructed[:min_len]

stoi_score = stoi(original, reconstructed, 16000)
pesq_score = pesq(16000, original, reconstructed, 'wb')

print("STOI:", stoi_score)
print("PESQ:", pesq_score)