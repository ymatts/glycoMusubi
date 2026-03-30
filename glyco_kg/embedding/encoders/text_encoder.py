"""Text encoder for knowledge graph embedding.

Encodes textual node attributes (disease names, pathway descriptions,
compound names, etc.) into fixed-dimensional vectors.

Two encoding strategies are provided:

* **hash_embedding** (default) -- deterministic text hashing into a
  learnable ``nn.Embedding`` table.  Fast, no external dependencies
  beyond PyTorch.
* **pubmedbert** -- frozen PubMedBERT backbone
  (``microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract``) with a
  trainable 2-layer MLP projection head (768 -> 384 -> 256).  Requires
  the ``transformers`` package.  BERT embeddings are pre-computed and
  cached at ``__init__`` time; the forward pass is a cheap index lookup
  followed by the MLP.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _stable_text_hash(text: str, num_buckets: int) -> int:
    """Return a deterministic bucket index for *text*.

    Uses SHA-256 for collision resistance and cross-platform stability
    (Python's built-in ``hash()`` is randomised across runs).
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return int(digest, 16) % num_buckets


class TextEncoder(nn.Module):
    """Encode text labels into fixed-dimensional embedding vectors.

    Parameters
    ----------
    num_entities : int
        Number of text-bearing entities in the KG.
    output_dim : int
        Dimensionality of the output embedding vector.
    method : str
        ``"hash_embedding"`` (default) or ``"pubmedbert"``.
    num_buckets : int or None
        Size of the hash-based embedding table (only for
        ``method="hash_embedding"``).  Defaults to
        ``min(num_entities * 2, 100_000)``.
    text_map : dict mapping int to str, optional
        Entity index -> text string mapping.  Required when
        *method* is ``"pubmedbert"``; ignored otherwise.
    pubmedbert_model_name : str
        HuggingFace model identifier for PubMedBERT.
    dropout : float
        Dropout probability in the PubMedBERT projection MLP.
    """

    VALID_METHODS = ("hash_embedding", "pubmedbert")

    def __init__(
        self,
        num_entities: int,
        output_dim: int = 256,
        method: str = "hash_embedding",
        num_buckets: int | None = None,
        text_map: Optional[Dict[int, str]] = None,
        pubmedbert_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; choose from {self.VALID_METHODS}"
            )

        self.num_entities = num_entities
        self.output_dim = output_dim
        self.method = method

        if method == "hash_embedding":
            self.num_buckets = num_buckets or min(num_entities * 2, 100_000)
            self.embedding = nn.Embedding(self.num_buckets, output_dim)
            nn.init.xavier_uniform_(self.embedding.weight)
            self.projection = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            )
        elif method == "pubmedbert":
            self._init_pubmedbert(
                num_entities=num_entities,
                output_dim=output_dim,
                text_map=text_map,
                model_name=pubmedbert_model_name,
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    # PubMedBERT helpers
    # ------------------------------------------------------------------

    def _init_pubmedbert(
        self,
        num_entities: int,
        output_dim: int,
        text_map: Optional[Dict[int, str]],
        model_name: str,
        dropout: float,
    ) -> None:
        """Set up PubMedBERT backbone, pre-compute embeddings, build MLP."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PubMedBERT text encoder requires the `transformers` "
                "package.  Install it with:  pip install transformers"
            ) from exc

        if text_map is None:
            raise ValueError(
                "text_map (Dict[int, str]) is required when method='pubmedbert'"
            )

        bert_dim = 768
        mid_dim = 384

        # Trainable 2-layer MLP: 768 -> 384 -> 256
        self.pubmedbert_mlp = nn.Sequential(
            nn.Linear(bert_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, output_dim),
        )

        # Pre-compute BERT embeddings (frozen; not stored as nn.Parameter)
        logger.info(
            "Loading PubMedBERT model '%s' to pre-compute %d text embeddings ...",
            model_name,
            len(text_map),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
        bert_model.eval()

        cached = torch.zeros(num_entities, bert_dim)
        texts_to_encode: List[int] = sorted(text_map.keys())

        # Batch encoding for efficiency
        batch_size = 64
        for start in range(0, len(texts_to_encode), batch_size):
            batch_indices = texts_to_encode[start : start + batch_size]
            batch_texts = [text_map[i] for i in batch_indices]

            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = bert_model(**tokens)
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]

            for j, idx in enumerate(batch_indices):
                cached[idx] = cls_embeddings[j]

        # Delete the BERT model to free memory -- only the MLP is trainable
        del bert_model, tokenizer

        # Register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("_pubmedbert_cache", cached)

        logger.info(
            "PubMedBERT embeddings cached for %d / %d entities.",
            len(texts_to_encode),
            num_entities,
        )

    # ------------------------------------------------------------------
    # Hash-embedding helpers
    # ------------------------------------------------------------------

    def text_to_index(self, text: str) -> int:
        """Map a text string to an embedding-table index (hash_embedding only)."""
        if self.method != "hash_embedding":
            raise RuntimeError("text_to_index is only available for method='hash_embedding'")
        return _stable_text_hash(text, self.num_buckets)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute text embeddings.

        Parameters
        ----------
        indices : torch.Tensor
            Long tensor of indices, shape ``[B]``.

            * For ``method="hash_embedding"``: bucket indices produced by
              :meth:`text_to_index`.
            * For ``method="pubmedbert"``: entity indices (keys from
              *text_map*).

        Returns
        -------
        torch.Tensor
            Shape ``[B, output_dim]``.
        """
        if self.method == "hash_embedding":
            emb = self.embedding(indices)
            return self.projection(emb)

        # PubMedBERT mode: look up cached embeddings, pass through MLP
        cached = self._pubmedbert_cache[indices]  # [B, 768]
        return self.pubmedbert_mlp(cached)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def encode_texts(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Convenience method: hash a list of strings and return embeddings.

        Only available for ``method="hash_embedding"``.

        Parameters
        ----------
        texts : list of str
            List of text strings to encode.
        device : torch.device, optional
            Target device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``[len(texts), output_dim]``.
        """
        if self.method != "hash_embedding":
            raise RuntimeError("encode_texts is only available for method='hash_embedding'")

        bucket_indices = torch.tensor(
            [self.text_to_index(t) for t in texts],
            dtype=torch.long,
        )
        if device is not None:
            bucket_indices = bucket_indices.to(device)
        else:
            # Use the device of the embedding parameters
            bucket_indices = bucket_indices.to(self.embedding.weight.device)
        return self.forward(bucket_indices)
