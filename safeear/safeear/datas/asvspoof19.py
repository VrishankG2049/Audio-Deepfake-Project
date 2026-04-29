from cProfile import label
import glob
import random
import os
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import torchaudio.functional

def get_path_iterator(tsv):
    """
    Get the root path and list of file lines from the TSV file.

    Args:
        tsv (str): Path to the TSV file.

    Returns:
        tuple: Root path and list of file lines.
    """
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
    return root, lines

def load_feature(feat_path):
    """
    Load feature from the specified path.

    Args:
        feat_path (str): Path to the feature file.

    Returns:
        np.ndarray: Loaded feature.
    """
    feat = np.load(feat_path, mmap_mode="r")
    return feat

class ASVSppof2019(Dataset):
    def __init__(self, tsv_path, protocol_path, feat_dir, max_len=64600, is_train=True):
        super().__init__()

        # ✅ Root directory (where .flac files are)
        self.root = Path(feat_dir)

        self.max_len = max_len
        self.is_train = is_train

        # ✅ Build dataset directly from protocol file (NO TSV)
        self.lines = []
        self.labels = {}

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                utt = parts[1]          # filename like LA_D_xxxxx
                label_str = parts[-1]   # bonafide / spoof

                if label_str == "bonafide":
                    label = 1
                elif label_str == "spoof":
                    label = 0
                else:
                    continue  # skip weird lines if any

                #self.lines.append(utt)
                #self.labels[utt] = label
                audio_path = self.root /  "flac" / (utt + ".flac")
                if not audio_path.exists():
                    raise RuntimeError(f"Missing file: {audio_path}")
                if audio_path.exists():
                    self.lines.append(utt)
                    self.labels[utt] = label

        # ✅ Get sample rate safely
        labels_tensor = torch.tensor(list(self.labels.values()))
        print("DATASET LABELS:", labels_tensor.unique())

        num_spoof = sum(1 for v in self.labels.values() if v == 0)
        num_bonafide = sum(1 for v in self.labels.values() if v == 1)

        print("SPOOF COUNT:", num_spoof)
        print("BONAFIDE COUNT:", num_bonafide)

        if len(self.lines) == 0:
            raise RuntimeError("Dataset is EMPTY — path is wrong")

        sample_file = self.lines[0]
        #sample_path = self.root / (sample_file + ".flac")
        sample_path = self.root /  "flac" / (sample_file + ".flac")
        _, self.sr = torchaudio.load(sample_path)

        # ✅ Debug (optional but useful)
        print("CHECK MAPPING:")
        for k in self.lines[:10]:
            print(k, self.labels[k])

        print("TOTAL SAMPLES:", len(self.lines))
        

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.lines)
    
    def __getitem__(self, idx):

        utt = self.lines[idx]

        #audio_path = os.path.join(self.root, utt + ".flac")
        audio_path = self.root / "flac" / (utt + ".flac")

        audio, sr = torchaudio.load(audio_path)

        label = self.labels[utt]

        return audio, label

    
    
def pad_sequence(batch):
    """
    Pad a sequence of tensors to have the same length.

    Args:
        batch (list of Tensors): List of tensors to pad.

    Returns:
        Tensor: Padded tensor with shape (batch_size, max_length, feature_dim).
    """
    batch = [item.permute(1, 0) for item in batch]  # Change shape from (feature_dim, length) to (length, feature_dim)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)  # Pad sequences
    return batch.permute(0, 2, 1)  # Change shape back to (batch_size, feature_dim, max_length)

def collate_fn(batch):

    wavs = []
    targets = []

    for wav, target in batch:

        # ensure shape = (1, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wavs.append(wav)
        targets.append(target)

    # pad along time dimension
    wavs = torch.nn.utils.rnn.pad_sequence(
        [w.squeeze(0) for w in wavs],  # (T,)
        batch_first=True
    )

    # restore channel dimension → (B, 1, T)
    wavs = wavs.unsqueeze(1)

    targets = torch.tensor(targets)

    return wavs, targets
class DataClass:
    def __init__(
        self,
        train_path, 
        val_path, 
        test_path, 
        max_len=64600,
    ) -> None:

        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_len = max_len

        # Get different datasets
        self.train = ASVSppof2019(
            self.train_path[0], 
            self.train_path[1], 
            self.train_path[2], 
            self.max_len, 
            is_train=True
        )
        self.val = ASVSppof2019(
            self.val_path[0], 
            self.val_path[1], 
            self.val_path[2], 
            self.max_len, 
            is_train=True
        )
        self.test = ASVSppof2019(
            self.test_path[0], 
            self.test_path[1], 
            self.test_path[2],
            self.max_len,
            is_train=False
        )
    def __call__(self, mode: str) -> ASVSppof2019:
        """Get dataset for a given mode.

        Args:
        ----
            mode (str): Mode of the dataset.

        Returns:
        -------
            ASVSppof2019: Dataset for the given mode.

        """
        if mode == "train":
            return self.train
        elif mode == "val":
            return self.val
        elif mode == "test":
            return self.test
        else:
            raise ValueError(f"Unknown mode: {mode}.")

class DataModule(LightningDataModule):
    def __init__(self, DataClass_dict, batch_size, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters(logger=False)
        DataClass_dict.pop("_target_")
        self.dataset_select = DataClass(**DataClass_dict)

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_select("train")
            self.data_val = self.dataset_select("val")
            self.data_test = self.dataset_select("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
