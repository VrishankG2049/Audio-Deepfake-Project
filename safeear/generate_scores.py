import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from safeear.trainer.safeear_trainer import SafeEarTrainer
from safeear.datas.asvspoof19 import DataModule

# -----------------------------
# LOAD CONFIG
# -----------------------------
cfg = OmegaConf.load("config/train19.yaml")

# -----------------------------
# BUILD MODELS
# -----------------------------
decouple_model = instantiate(cfg.decouple_model)
detect_model = instantiate(cfg.detect_model)

# -----------------------------
# LOAD TRAINED SYSTEM
# -----------------------------
system = SafeEarTrainer.load_from_checkpoint(
    "Exps/ASVspoof19/checkpoints/epoch=0-val_eer=0.3976.ckpt",
    decouple_model=decouple_model,
    detect_model=detect_model,
    lr_raw_former=cfg.system.lr_raw_former,
    save_score_path=cfg.system.save_score_path
)

system = system.cuda()
system.eval()

# -----------------------------
# LOAD DATA (CORRECT WAY)
# -----------------------------
from hydra.utils import instantiate

datamodule = instantiate(cfg.datamodule)
datamodule.setup()

loader = datamodule.val_dataloader()

# -----------------------------
# OUTPUT FILE
# -----------------------------
out_file = open("scores.txt", "w")

# -----------------------------
# GENERATE SCORES
# -----------------------------
for batch in loader:
    x, target = batch

    x = x.cuda()
    target = target.cuda()

    with torch.no_grad():
        _, _, score, target = system.forward(
            (x, target),
            is_train=False
        )

    score = score.cpu().numpy()
    target = target.cpu().numpy()

    for i in range(len(score)):
        utt = f"utt_{i}"
        label = "bonafide" if target[i] == 0 else "spoof"

        out_file.write(f"{utt} {score[i]} {label}\n")

out_file.close()

print("Scores saved to scores.txt")