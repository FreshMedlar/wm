"""
First-Principles Active Inference Agent with Single Shared Encoder and Target Detach.
"""

from FPagent.agent import FirstPrinciplesActiveInferenceAgent
from FPagent.buffer import PixelReplayBuffer
from FPagent.env import GymPixelWrapper, make_pixel_env
from FPagent.losses import SIGReg, compute_latent_metrics, sigreg_loss
from FPagent.networks import (
    PixelEncoder,
    PredictorEnsemble,
    RewardPredictor,
    TransitionPredictor,
    ViabilityClassifier,
)
from FPagent.planner import ActiveInferencePlanner

__all__ = [
    "FirstPrinciplesActiveInferenceAgent",
    "PixelReplayBuffer",
    "GymPixelWrapper",
    "make_pixel_env",
    "SIGReg",
    "sigreg_loss",
    "compute_latent_metrics",
    "PixelEncoder",
    "TransitionPredictor",
    "PredictorEnsemble",
    "ViabilityClassifier",
    "RewardPredictor",
    "ActiveInferencePlanner",
]
