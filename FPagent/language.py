"""
Language Context Encoder for First-Principles Active Inference Agent.

Provides a frozen Transformer text encoder (BERT / MiniLM) with a lightweight,
trainable projection adapter mapping textual instructions into a compact context embedding.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LanguageContextEncoder(nn.Module):
    """
    Frozen pretrained language encoder with a trainable projection head.
    Outputs a context vector c_emb in R^{context_dim} used to modulate
    world model transitions, viability boundaries, and pragmatic goals.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        context_dim: int = 64,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.context_dim = context_dim

        # Load tokenizer and backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)

        # Freeze the transformer backbone completely
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

        bert_hidden_dim = self.bert.config.hidden_size

        # Trainable adapter: projects high-dim BERT space -> world model context space
        self.projector = nn.Sequential(
            nn.Linear(bert_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, context_dim),
            nn.LayerNorm(context_dim),
        ).to(self.device)

    @torch.no_grad()
    def _extract_cls(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Tokenize and extract pooled representation from frozen transformer."""
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.bert(**tokens)
        # Use mean pooling with attention mask for robust sentence embeddings
        attention_mask = tokens["attention_mask"].unsqueeze(-1)  # (B, L, 1)
        hidden = outputs.last_hidden_state  # (B, L, H)
        sum_hidden = torch.sum(hidden * attention_mask, dim=1)  # (B, H)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        return pooled

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Args:
            text: A single string or list of B strings.

        Returns:
            c_emb: (B, context_dim) tensor.
        """
        pooled = self._extract_cls(text)
        return self.projector(pooled)
