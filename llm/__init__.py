"""
Hierarchical LeJEPA Language Model Package.
Conditioning the autoregressive decoder on both past tokens and predicted latents:
P(x_{t+1} | x_{\le t}, z_hat_{t+1})
"""

from llm.models import (
    LeJEPALanguageModel,
    LeJEPAEncoder,
    LeJEPAPredictor,
    ConditionalDecoder,
    StandardTransformerLM,
)
from llm.modules import SIGReg, CausalSelfAttention, TransformerBlock, ConditionalTransformerBlock
from llm.dataset import MemmapTextDataset, get_dataloaders

__all__ = [
    "LeJEPALanguageModel",
    "LeJEPAEncoder",
    "LeJEPAPredictor",
    "ConditionalDecoder",
    "StandardTransformerLM",
    "SIGReg",
    "CausalSelfAttention",
    "TransformerBlock",
    "ConditionalTransformerBlock",
    "MemmapTextDataset",
    "get_dataloaders",
]
