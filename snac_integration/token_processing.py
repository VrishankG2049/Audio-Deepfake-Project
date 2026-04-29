import torch

def tokens_to_features(tokens):
    """
    tokens: list of tensors [T1, T2, T3, T4]
            each of shape (B, N_tokens)

    returns: list of tensors [(B, 1, T1), (B, 1, T2), ...]
    """

    features = []

    for t in tokens:
        # convert to float
        t = t.float()

        # add channel dimension
        t = t.unsqueeze(1)  # (B, 1, T)

        features.append(t)

    return features
import torch.nn.functional as F

def prepare_acoustic_features(features, mode="T3_T4"):
    """
    features: list of tensors [(B,1,T1), (B,1,T2), (B,1,T3), (B,1,T4)]

    returns: (B, C, T) ready for detector
    """

    if mode == "T4":
        return features[-1]

    elif mode == "T3_T4":
        t3 = features[-2]
        t4 = features[-1]

        # Upsample t3 → match t4 length
        t3 = F.interpolate(t3, size=t4.shape[-1], mode="nearest")

        # Concatenate along channel dimension
        out = torch.cat([t3, t4], dim=1)

        return out

    elif mode == "T2_T3_T4":
        t2, t3, t4 = features[-3], features[-2], features[-1]

        target_len = t4.shape[-1]

        t2 = F.interpolate(t2, size=target_len, mode="nearest")
        t3 = F.interpolate(t3, size=target_len, mode="nearest")

        out = torch.cat([t2, t3, t4], dim=1)

        return out

    else:
        raise ValueError("Invalid mode")
