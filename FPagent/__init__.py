"""
First-Principles Active Inference Agent with Single Shared Encoder and Target Detach.
"""

from firstPAgent.agent import FirstPrinciplesActiveInferenceAgent
from firstPAgent.buffer import PixelReplayBuffer
from firstPAgent.env import GymPixelWrapper, make_pixel_env
from firstPAgent.losses import SIGReg, compute_latent_metrics, sigreg_loss
from firstPAgent.networks import (
    PixelEncoder,
    PredictorEnsemble,
    RewardPredictor,
    TransitionPredictor,
    ViabilityClassifier,
)
from firstPAgent.planner import ActiveInferencePlanner

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
