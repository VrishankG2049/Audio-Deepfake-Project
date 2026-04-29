# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""
import torch.nn as nn
from einops import rearrange
import torch

from .modules.seanet import SEANetEncoder, SEANetDecoder
from .modules.quantization  import ResidualVectorQuantizer

import sys
import os

# Adjust path to your project root
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
import sys
import os

# Get absolute path to project root (safe_snac)
CURRENT_DIR = os.path.dirname(__file__)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from snac_integration.snac_wrapper import SNACWrapper
from snac_integration.token_processing import tokens_to_features, prepare_acoustic_features

class SpeechTokenizer(nn.Module):
    def __init__(self, n_filters, dimension, strides, lstm_layers, bidirectional, dilation_base, residual_kernel_size, n_residual_layers, activation, sample_rate, n_q, semantic_dimension, codebook_size):
        '''
        
        Parameters
        ----------
        n_filters : int
            Number of filters in the SEANet encoder/decoder.
        dimension : int
            Dimensionality of the encoder/decoder.
        strides : list
            List of stride values for the SEANet encoder/decoder.
        lstm_layers : int
            Number of LSTM layers in the encoder/decoder.
        bidirectional : bool
            Whether to use bidirectional LSTM in the encoder.
        dilation_base : int
            Base dilation rate for the residual blocks in the encoder/decoder.
        residual_kernel_size : int
            Kernel size for the residual blocks in the encoder/decoder.
        n_residual_layers : int
            Number of residual layers in the encoder/decoder.
        activation : str
            Activation function to use in the encoder/decoder.
        sample_rate : int
            Sample rate of the audio.
        n_q : int
            Number of quantization levels.
        semantic_dimension : int
            Dimensionality of the semantic representation.
        codebook_size : int
            Size of the codebook for vector quantization.

        '''
        super().__init__()
        self.encoder = SEANetEncoder(n_filters=n_filters, 
                                     dimension=dimension, 
                                     ratios=strides,
                                     lstm=lstm_layers,
                                     bidirectional=bidirectional,
                                     dilation_base=dilation_base,
                                     residual_kernel_size=residual_kernel_size,
                                     n_residual_layers=n_residual_layers,
                                     activation=activation)
        self.sample_rate = sample_rate
        self.n_q = n_q
        if dimension != semantic_dimension:
            self.transform = nn.Linear(dimension, semantic_dimension)
        else:
            self.transform = nn.Identity()
        self.quantizer = ResidualVectorQuantizer(dimension=dimension, n_q=n_q, bins=codebook_size)
        self.decoder = SEANetDecoder(n_filters=n_filters, 
                                     dimension=dimension, 
                                     ratios=strides,
                                     lstm=lstm_layers,
                                     bidirectional=False,
                                     dilation_base=dilation_base,
                                     residual_kernel_size=residual_kernel_size,
                                     n_residual_layers=n_residual_layers,
                                     activation=activation)
        self.snac = SNACWrapper(device="cuda")
        self.snac_proj = nn.Linear(1024, 768)
    @classmethod
    def load_from_checkpoint(cls, 
                             config_path: str, 
                             ckpt_path: str):
        '''

        Parameters
        ----------
        config_path : str
            Path of model configuration file.
        ckpt_path : str
            Path of model  checkpoint.

        Returns
        -------
        model : SpeechTokenizer
            SpeechTokenizer model.

        '''
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(cfg)
        params = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(params)
        return model
    
    
    def forward(self, 
            x: torch.tensor, 
            n_q: int=None, 
            layers: list=[0]):

        """
        SNAC-based forward (FINAL CORRECT VERSION)
        """

        #print("[DEBUG] USING SNAC FORWARD")

    # -------------------------
    # 1. SNAC encoding
    # -------------------------
        tokens = self.snac.encode(x)
    # tokens = [T1, T2, T3, T4]

    # -------------------------
    # 2. Build multi-scale features (IMPORTANT)
    # -------------------------
        features_list = []

    # use largest temporal resolution
        target_len = tokens[-1].shape[-1]

        for t in tokens:
        # ensure (B, C, T)
            if t.dim() == 2:
                t = t.unsqueeze(1)
            t=t.float()
            t = t.repeat(1, 1024, 1)   # (B, 1024, T)

        # align time dimension
            if t.shape[-1] != target_len:
                t = torch.nn.functional.interpolate(
                    t,
                    size=target_len,
                    mode='nearest'
                )

            features_list.append(t)
        # 🔥 FORCE EXACTLY 7 SCALES
        if len(features_list) > 7:
            features_list = features_list[:7]
        while len(features_list) < 7:
            features_list.append(features_list[-1])

    # -------------------------
    # 3. Create feature for classifier
    # -------------------------
    # use last (highest resolution)
        feature = features_list[-1]              # (B, C, T)
        
        

        feature = feature.transpose(1, 2)       # (B, T, C)

    # project to SafeEar dim (768)
        feature = self.snac_proj(feature)

    # transformer
        #feature = self.transform(feature)

    # -------------------------
    # 4. Dummy outputs (keep interface)
    # -------------------------
        o = x
        commit_loss = torch.tensor(0.0, device=x.device)

    # IMPORTANT: return multi-scale features
        return o, commit_loss, feature, features_list
    
    def forward_feature(self, x: torch.tensor, layers: list=None):

        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape should be (batch, channels, timesteps).
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is all layers.

        Returns
        -------
        quantized_list : list[torch.tensor]
            Quantized of required layers.
        # SNAC encoding
        tokens = self.snac.encode(x)

        # tokens → features
        features_list = tokens_to_features(tokens)

        # acoustic features (T3 + T4)
        acoustic = prepare_acoustic_features(features_list, mode="T3_T4")

        return [acoustic]

        '''
        e = self.encoder(x)
        layers = layers if layers else list(range(self.n_q))
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, layers=layers)
        return quantized_list
    
    def encode(self, 
               x: torch.tensor, 
               n_q: int=None, 
               st: int=None):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (n_q, batch, timesteps)

        '''
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        codes = self.quantizer.encode(e, n_q=n_q, st=st)
        return codes
    
    def decode(self, 
               codes: torch.tensor, 
               st: int=0):
        '''

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        '''
        quantized = self.quantizer.decode(codes, st=st)
        o = self.decoder(quantized)
        return o
