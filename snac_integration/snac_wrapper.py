import torch
import sys
import os

# Add SNAC repo to path
SNAC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../snac"))
sys.path.append(SNAC_PATH)

# Import SNAC
try:
    from snac import SNAC
except:
    try:
        from model import SNAC
    except:
        raise ImportError("Could not import SNAC model. Check repo structure.")

class SNACWrapper:
    def __init__(self, device="cuda"):
        self.device = device

        print("[INFO] Loading SNAC model...")
        self.model = SNAC().to(device)
        self.model.eval()
        print("[INFO] SNAC model loaded.")

    def encode(self, audio):
        audio = audio.to(self.device)

        with torch.no_grad():
            tokens = self.model.encode(audio)
        

        return tokens
