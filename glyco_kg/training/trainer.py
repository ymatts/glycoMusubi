"""KGE model training loop with mixed-precision, callbacks, and checkpointing."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from glycoMusubi.embedding.models.base import BaseKGEModel
from glycoMusubi.training.callbacks import Callback

logger = logging.getLogger(__name__)
EdgeType = Tuple[str, str, str]


class Trainer:
    """Training loop for heterogeneous KGE models.

    Supports mixed-precision training (BF16/FP16), callback-driven early
    stopping / checkpointing, and optional validation between epochs.

    Parameters
    ----------
    model : BaseKGEModel
        The KGE model to train.
    optimizer : Optimizer
        PyTorch optimiser (e.g. ``Adam``).
    loss_fn : nn.Module
        Loss function accepting ``(pos_scores, neg_scores) -> scalar``.
    train_data : HeteroData
        Training split of the heterogeneous graph.
    val_data : HeteroData or None
        Validation split (if ``None``, no validation is performed).
    negative_sampler : callable or None
        Function ``(batch) -> neg_scores`` that produces negative samples.
        When ``None``, the caller must supply ``neg_scores`` externally.
    scheduler : _LRScheduler or str or None
        Learning-rate scheduler.  Can be an instantiated ``_LRScheduler``,
        a string shortcut (``"cosine_warm_restarts"``), or ``None`` to
        use the default ``CosineAnnealingWarmRestarts(T_0=10, T_mult=2)``.
        Pass ``"none"`` to explicitly disable scheduling.
    callbacks : Sequence[Callback]
        List of :class:`Callback` instances.
    device : str or torch.device
        Target device (default ``'cpu'``).
    mixed_precision : bool
        Enable automatic mixed precision (default ``False``).
    amp_dtype : torch.dtype
        Dtype for the AMP context (``torch.float16`` or ``torch.bfloat16``).
    grad_clip_norm : float or None
        Max gradient norm for clipping (``None`` to disable).
    use_hgt_loader : bool
        Enable HGTLoader-based mini-batch training (default ``False``).
        When ``True``, overrides ``use_mini_batch`` and uses
        ``hgt_num_samples`` / ``hgt_batch_size``.
    hgt_num_samples : list of int or None
        Neighbours per edge type per layer for HGTLoader
        (default ``[15]``).
    hgt_batch_size : int
        Target number of seed nodes per HGTLoader batch (default 1024).
    gradient_accumulation_steps : int
        Accumulate gradients over this many mini-batches before
        performing an optimiser step (default 1).
    relation_balance_alpha : float
        Relation-balancing strength in ``[0, 1]``. 0 disables balancing.
        Positive values upweight sparse edge types using
        ``(max_count / rel_count) ** alpha``.
    relation_balance_max_weight : float
        Upper cap for per-relation weights before normalisation.
    """

    def __init__(
        self,
        model: BaseKGEModel,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        train_data: HeteroData,
        val_data: Optional[HeteroData] = None,
        negative_sampler: Optional[Callable[..., torch.Tensor]] = None,
        scheduler: Optional[Union[_LRScheduler, str]] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        device: Union[str, torch.device] = "cpu",
        mixed_precision: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        grad_clip_norm: Optional[float] = None,
        use_mini_batch: bool = False,
        mini_batch_size: int = 1024,
        num_samples: Optional[List[int]] = None,
        use_hgt_loader: bool = False,
        hgt_num_samples: Optional[List[int]] = None,
        hgt_batch_size: int = 1024,
        gradient_accumulation_steps: int = 1,
        relation_balance_alpha: float = 0.0,
        relation_balance_max_weight: float = 5.0,
        max_edges_per_type: int = 0,
        num_negatives: int = 1,
        neg_pool_restrictor: Optional[Callable[[Tuple[str, str, str]], Optional[torch.Tensor]]] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device) if val_data is not None else None
        self.negative_sampler = negative_sampler
        self.num_negatives = max(1, num_negatives)
        self.scheduler = self._resolve_scheduler(scheduler, optimizer)
        self.callbacks: List[Callback] = list(callbacks) if callbacks else []
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        self.amp_dtype = amp_dtype
        self.grad_clip_norm = grad_clip_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.relation_balance_alpha = max(0.0, float(relation_balance_alpha))
        self.relation_balance_max_weight = max(1.0, float(relation_balance_max_weight))
        self.max_edges_per_type = max(0, max_edges_per_type)
        self.neg_pool_restrictor = neg_pool_restrictor

        # Per-edge-type gradient accumulation for memory-heavy decoders
        self._per_type_backprop = (
            hasattr(model, "_decoder_type")
            and model._decoder_type == "hybrid"
        )

        # HGTLoader support: use_hgt_loader takes precedence over use_mini_batch
        if use_hgt_loader:
            self.use_mini_batch = True
            self.mini_batch_size = hgt_batch_size
            self.num_samples = hgt_num_samples if hgt_num_samples is not None else [15]
        else:
            self.use_mini_batch = use_mini_batch
            self.mini_batch_size = mini_batch_size
            self.num_samples = num_samples if num_samples is not None else [512, 256]

        # GradScaler only needed for float16 on CUDA
        self._scaler: Optional[GradScaler] = None
        if mixed_precision and amp_dtype == torch.float16 and self.device.type == "cuda":
            self._scaler = GradScaler()

        self.current_epoch = 0

        # Create HGTLoader if mini-batch training is enabled
        self._hgt_loader = None
        if self.use_mini_batch:
            self._hgt_loader = self._create_hgt_loader(
                self.train_data, self.num_samples, self.mini_batch_size
            )

        self._edge_type_weights = self._build_edge_type_weights(self.train_data)
        if self._edge_type_weights:
            preview = sorted(
                self._edge_type_weights.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
            preview_str = ", ".join(f"{k}:{v:.2f}" for k, v in preview)
            logger.info("Relation-balanced training enabled (top weights): %s", preview_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int,
        validate_every: int = 1,
        val_score_fn: Optional[Callable[[BaseKGEModel, HeteroData], Dict[str, float]]] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Parameters
        ----------
        epochs : int
            Maximum number of training epochs.
        validate_every : int
            Run validation every N epochs (default 1).
        val_score_fn : callable or None
            ``(model, val_data) -> {metric_name: value}`` used during
            validation.  If ``None`` and ``val_data`` is set, a simple
            loss-based validation is performed.

        Returns
        -------
        Dict[str, List[float]]
            History dict with keys like ``'train_loss'``, ``'val_mrr'``, etc.
        """
        history: Dict[str, List[float]] = {"train_loss": []}

        self._fire_on_train_begin()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._fire_on_epoch_begin(epoch)

            train_loss = self.train_epoch()
            history["train_loss"].append(train_loss)

            # Validation
            val_metrics: Optional[Dict[str, float]] = None
            if self.val_data is not None and epoch % validate_every == 0:
                if val_score_fn is not None:
                    val_metrics = self.validate(val_score_fn)
                else:
                    val_metrics = self._validate_loss()

                for k, v in val_metrics.items():
                    history.setdefault(f"val_{k}", []).append(v)

            # LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Callbacks (epoch end)
            self._fire_on_epoch_end(epoch, train_loss, val_metrics)

            # Check early stopping
            if self._should_stop():
                logger.info("Stopping training at epoch %d", epoch)
                break

        self._fire_on_train_end()
        return history

    def train_epoch(self) -> float:
        """Run one epoch of training over the full training data.

        When ``use_mini_batch`` is enabled, iterates over subgraphs
        sampled by :class:`~torch_geometric.loader.HGTLoader`.
        Otherwise falls back to full-batch training.

        Returns
        -------
        float
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        if self.use_mini_batch and self._hgt_loader is not None:
            accum_steps = self.gradient_accumulation_steps
            self.optimizer.zero_grad()

            for step, batch_data in enumerate(self._hgt_loader, 1):
                batch_data = batch_data.to(self.device)
                grouped_scores = self._compute_scores_grouped(batch_data)
                loss = self._accumulation_step(
                    grouped_scores, accum_steps
                )
                total_loss += loss
                n_batches += 1

                # Perform optimiser step after accumulation_steps
                if step % accum_steps == 0:
                    self._optimizer_step()
                    self.optimizer.zero_grad()

            # Handle remaining accumulated gradients
            if n_batches % accum_steps != 0:
                self._optimizer_step()
        elif self._per_type_backprop:
            # Per-edge-type gradient accumulation: process one edge type at
            # a time to keep peak GPU memory low for heavy decoders.
            total_loss, n_batches = self._train_epoch_per_edge_type()
        else:
            # Simple full-batch training over the heterogeneous graph.
            grouped_scores = self._compute_scores_grouped(self.train_data)
            loss = self._training_step(grouped_scores)
            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _train_epoch_per_edge_type(self) -> Tuple[float, int]:
        """Train one epoch, processing edge types one at a time.

        For each edge type:
          1. Run ``model.forward(data)`` once to get embeddings.
          2. Split edges into sub-batches to bound peak GPU memory.
          3. Each sub-batch: score, compute loss, backward.
             Uses ``retain_graph=True`` to keep the forward graph alive
             across sub-batches within the same edge type.
        After all edge types: perform a single optimiser step.

        Returns
        -------
        Tuple[float, int]
            ``(total_loss, n_batches)``
        """
        self.optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        K = self.num_negatives
        # Sub-batch size: keep scorer calls bounded per backward pass.
        # K=1: process full edge type at once (fast, like original R1).
        # K>1: split into sub-batches of mini_batch_size to control memory.
        sub_batch = (
            self.max_edges_per_type if self.max_edges_per_type > 0 else 999_999_999
        ) if K <= 1 else self.mini_batch_size

        _use_rel_idx = (
            hasattr(self.model, "_decoder_type")
            and self.model._decoder_type == "hybrid"
        )

        for edge_type, edge_store in self.train_data.edge_items():
            src_type, _, dst_type = edge_type
            edge_index = edge_store.edge_index
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue

            # Subsample if needed
            if self.max_edges_per_type > 0 and num_edges > self.max_edges_per_type:
                perm = torch.randperm(num_edges, device=edge_index.device)[
                    : self.max_edges_per_type
                ]
                edge_index = edge_index[:, perm]
                if hasattr(edge_store, "edge_type_idx"):
                    rel_idx_full = edge_store.edge_type_idx[perm]
                else:
                    rel_idx_full = None
            else:
                rel_idx_full = (
                    edge_store.edge_type_idx
                    if hasattr(edge_store, "edge_type_idx")
                    else None
                )

            n_edges = edge_index.size(1)
            weight = float(self._edge_type_weights.get(edge_type, 1.0))

            # One forward pass per edge type (embedding lookups + fusion)
            emb_dict = self.model(self.train_data)
            dst_pool = emb_dict[dst_type]
            # Restrict negative pool if applicable (e.g. function-aware negatives)
            if self.neg_pool_restrictor is not None:
                restricted_idx = self.neg_pool_restrictor(edge_type)
                if restricted_idx is not None:
                    restricted_idx = restricted_idx.to(dst_pool.device)
                    dst_pool = dst_pool[restricted_idx]
            src_pool = emb_dict[src_type]

            # Process edges in sub-batches
            n_sub = max(1, (n_edges + sub_batch - 1) // sub_batch)
            for sub_idx in range(n_sub):
                start = sub_idx * sub_batch
                end = min(start + sub_batch, n_edges)
                ei_batch = edge_index[:, start:end]

                if rel_idx_full is not None:
                    ri_batch = rel_idx_full[start:end]
                else:
                    ri_batch = torch.zeros(
                        end - start, dtype=torch.long, device=self.device
                    )

                head_emb = emb_dict[src_type][ei_batch[0]]
                tail_emb = emb_dict[dst_type][ei_batch[1]]

                if _use_rel_idx:
                    rel_for_score = ri_batch
                else:
                    rel_for_score = self.model.get_relation_embedding(ri_batch)

                _amp_ctx = (
                    autocast(device_type=self.device.type, dtype=self.amp_dtype)
                    if self.mixed_precision
                    else contextlib.nullcontext()
                )
                with _amp_ctx:
                    pos_scores = self.model.score(
                        head_emb, rel_for_score, tail_emb
                    )

                    if self.negative_sampler is not None:
                        neg_scores = self.negative_sampler(
                            head_emb, rel_for_score, tail_emb, emb_dict,
                            edge_type,
                        )
                    else:
                        neg_scores = self._sample_negatives(
                            head_emb, rel_for_score, dst_pool, K,
                            src_pool=src_pool, tail_emb=tail_emb,
                        )

                    loss = self.loss_fn(pos_scores, neg_scores) * weight

                # Retain graph for all but the last sub-batch of this edge type
                is_last = sub_idx == n_sub - 1
                if self.mixed_precision and self._scaler is not None:
                    self._scaler.scale(loss).backward(
                        retain_graph=not is_last
                    )
                else:
                    loss.backward(retain_graph=not is_last)

                total_loss += loss.item()
                n_batches += 1

        # Single optimiser step after all edge types
        if self.grad_clip_norm is not None:
            if self._scaler is not None:
                self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        return total_loss, n_batches

    def validate(
        self,
        val_score_fn: Callable[[BaseKGEModel, HeteroData], Dict[str, float]],
    ) -> Dict[str, float]:
        """Run validation using a user-supplied scoring function.

        Parameters
        ----------
        val_score_fn : callable
            ``(model, val_data) -> {metric: value}``

        Returns
        -------
        Dict[str, float]
            Validation metrics.
        """
        self.model.eval()
        with torch.no_grad():
            metrics = val_score_fn(self.model, self.val_data)
        return metrics

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist model, optimiser, and scheduler state.

        Parameters
        ----------
        path : str or Path
            File path for the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self._scaler is not None:
            state["scaler_state_dict"] = self._scaler.state_dict()
        torch.save(state, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore model, optimiser, and scheduler state.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path.
        """
        # weights_only=False is required here because the checkpoint contains
        # optimizer state_dict and scheduler state_dict (not just model weights).
        state = torch.load(path, map_location=self.device, weights_only=False)  # noqa: S614
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_epoch = state.get("epoch", 0)
        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self._scaler is not None and "scaler_state_dict" in state:
            self._scaler.load_state_dict(state["scaler_state_dict"])
        logger.info("Checkpoint loaded from %s (epoch %d)", path, self.current_epoch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_scheduler(
        scheduler: Optional[Union[_LRScheduler, str]],
        optimizer: Optimizer,
    ) -> Optional[_LRScheduler]:
        """Resolve a scheduler specification into an ``_LRScheduler`` instance.

        Parameters
        ----------
        scheduler : _LRScheduler, str, or None
            - ``None`` -- use default ``CosineAnnealingWarmRestarts(T_0=10, T_mult=2)``.
            - ``"none"`` -- disable scheduling (returns ``None``).
            - ``"cosine_warm_restarts"`` -- create ``CosineAnnealingWarmRestarts``.
            - An ``_LRScheduler`` instance -- use as-is.
        optimizer : Optimizer
            The optimiser to attach the scheduler to.

        Returns
        -------
        _LRScheduler or None
        """
        if scheduler is None:
            return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        if isinstance(scheduler, str):
            name = scheduler.lower().replace("-", "_")
            if name == "none":
                return None
            if name in ("cosine_warm_restarts", "cosine_annealing_warm_restarts"):
                return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            raise ValueError(
                f"Unknown scheduler name {scheduler!r}; "
                "choose from 'cosine_warm_restarts' or 'none'."
            )
        # Assume it's an instantiated scheduler -- use as-is.
        return scheduler

    def _create_hgt_loader(
        self,
        data: HeteroData,
        num_samples: List[int],
        batch_size: int,
    ) -> Any:
        """Create a PyG loader for mini-batch subgraph sampling.

        Tries HGTLoader first (requires torch-sparse).  Falls back to
        NeighborLoader which is built into PyG core.

        Parameters
        ----------
        data : HeteroData
            The full heterogeneous graph.
        num_samples : list of int
            Number of neighbours to sample at each GNN layer
            (e.g. ``[512, 256]``).
        batch_size : int
            Number of seed nodes per mini-batch.

        Returns
        -------
        HGTLoader or NeighborLoader or None
        """
        # Build input_nodes: sample from the largest node type
        input_node_type: Optional[str] = None
        max_nodes = 0
        for nt in data.node_types:
            num_nodes = data[nt].num_nodes
            if num_nodes is not None and num_nodes > max_nodes:
                max_nodes = num_nodes
                input_node_type = nt

        if input_node_type is None:
            logger.warning("No node types with nodes found; loader not created.")
            return None

        # Try HGTLoader first
        try:
            from torch_geometric.loader import HGTLoader
            loader = HGTLoader(
                data,
                num_samples=num_samples,
                batch_size=batch_size,
                input_nodes=(input_node_type, None),
                shuffle=True,
            )
            logger.info(
                "Created HGTLoader: input_nodes=%s, num_samples=%s, batch_size=%d",
                input_node_type,
                num_samples,
                batch_size,
            )
            return loader
        except (ImportError, Exception) as e:
            logger.warning("HGTLoader unavailable (%s), falling back to NeighborLoader", e)

        # Fallback: NeighborLoader (no torch-sparse needed)
        from torch_geometric.loader import NeighborLoader
        loader = NeighborLoader(
            data,
            num_neighbors=num_samples,
            batch_size=batch_size,
            input_nodes=input_node_type,
            shuffle=True,
        )
        logger.info(
            "Created NeighborLoader: input_nodes=%s, num_neighbors=%s, batch_size=%d",
            input_node_type,
            num_samples,
            batch_size,
        )
        return loader

    def _sample_negatives(
        self,
        head_emb: torch.Tensor,
        rel: torch.Tensor,
        dst_pool: torch.Tensor,
        K: int,
        max_score_batch: int = 200_000,
        src_pool: Optional[torch.Tensor] = None,
        tail_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample K negatives per positive with bidirectional corruption.

        When ``src_pool`` and ``tail_emb`` are provided, half the negatives
        corrupt the tail (standard) and half corrupt the head. This ensures
        both head and tail embeddings are trained symmetrically.

        When K=1, returns [B].  When K>1, returns [B, K].
        Scoring is done in chunks to avoid OOM.
        """
        n_edges = head_emb.size(0)
        num_dst = dst_pool.size(0)

        if K == 1:
            # K=1: randomly corrupt head or tail with 50/50 probability
            if src_pool is not None and tail_emb is not None:
                num_src = src_pool.size(0)
                corrupt_head = torch.rand(n_edges, device=self.device) < 0.5
                # Tail corruption
                rand_tail = torch.randint(0, num_dst, (n_edges,), device=self.device)
                neg_tail = dst_pool[rand_tail]
                # Head corruption
                rand_head = torch.randint(0, num_src, (n_edges,), device=self.device)
                neg_head = src_pool[rand_head]
                # Mix: where corrupt_head, use (neg_head, tail); else (head, neg_tail)
                final_head = torch.where(corrupt_head.unsqueeze(-1), neg_head, head_emb)
                final_tail = torch.where(corrupt_head.unsqueeze(-1), tail_emb, neg_tail)
                return self.model.score(final_head, rel, final_tail)
            else:
                rand_idx = torch.randint(0, num_dst, (n_edges,), device=self.device)
                neg_tail_emb = dst_pool[rand_idx]
                return self.model.score(head_emb, rel, neg_tail_emb)

        # K > 1: split into K_tail (tail corruption) and K_head (head corruption)
        bidirectional = src_pool is not None and tail_emb is not None
        if bidirectional:
            K_tail = K // 2
            K_head = K - K_tail
        else:
            K_tail = K
            K_head = 0

        all_scores = []
        D = head_emb.size(-1)

        # --- Tail corruption: (head, rel, random_tail) ---
        if K_tail > 0:
            rand_idx = torch.randint(0, num_dst, (n_edges, K_tail), device=self.device)
            neg_tail_flat = dst_pool[rand_idx.view(-1)]
            head_flat = head_emb.unsqueeze(1).expand(n_edges, K_tail, D).reshape(-1, D)
            if rel.dim() == 1:
                rel_flat = rel.unsqueeze(1).expand(n_edges, K_tail).reshape(-1)
            else:
                rel_flat = rel.unsqueeze(1).expand(n_edges, K_tail, rel.size(-1)).reshape(-1, rel.size(-1))

            tail_scores = self._score_in_chunks(head_flat, rel_flat, neg_tail_flat, max_score_batch)
            all_scores.append(tail_scores.view(n_edges, K_tail))

        # --- Head corruption: (random_head, rel, tail) ---
        if K_head > 0:
            num_src = src_pool.size(0)
            rand_idx = torch.randint(0, num_src, (n_edges, K_head), device=self.device)
            neg_head_flat = src_pool[rand_idx.view(-1)]
            tail_flat = tail_emb.unsqueeze(1).expand(n_edges, K_head, D).reshape(-1, D)
            if rel.dim() == 1:
                rel_flat = rel.unsqueeze(1).expand(n_edges, K_head).reshape(-1)
            else:
                rel_flat = rel.unsqueeze(1).expand(n_edges, K_head, rel.size(-1)).reshape(-1, rel.size(-1))

            head_scores = self._score_in_chunks(neg_head_flat, rel_flat, tail_flat, max_score_batch)
            all_scores.append(head_scores.view(n_edges, K_head))

        return torch.cat(all_scores, dim=-1)  # [B, K]

    def _score_in_chunks(
        self,
        head: torch.Tensor,
        rel: torch.Tensor,
        tail: torch.Tensor,
        max_batch: int,
    ) -> torch.Tensor:
        """Score (head, rel, tail) triples in memory-bounded chunks."""
        total = head.size(0)
        if total <= max_batch:
            return self.model.score(head, rel, tail)
        chunks = []
        for i in range(0, total, max_batch):
            j = min(i + max_batch, total)
            c = self.model.score(head[i:j], rel[i:j], tail[i:j])
            chunks.append(c)
        return torch.cat(chunks)

    def _build_edge_type_weights(self, data: HeteroData) -> Dict[EdgeType, float]:
        """Compute per-edge-type weights for relation-balanced training.

        Weights are based on edge counts in the training split:
        ``w_r = (max_count / count_r) ** alpha``.
        They are then capped and normalised to mean 1.0.
        """
        if self.relation_balance_alpha <= 0.0:
            return {}

        counts: Dict[EdgeType, int] = {}
        for etype in data.edge_types:
            n_edges = int(data[etype].edge_index.size(1))
            if n_edges > 0:
                counts[etype] = n_edges

        if not counts:
            return {}

        max_count = max(counts.values())
        raw: Dict[EdgeType, float] = {}
        for etype, count in counts.items():
            w = (max_count / float(count)) ** self.relation_balance_alpha
            raw[etype] = min(w, self.relation_balance_max_weight)

        mean_w = sum(raw.values()) / float(len(raw))
        if mean_w <= 0:
            return {}

        return {etype: (w / mean_w) for etype, w in raw.items()}

    def _compute_scores_grouped(
        self, data: HeteroData
    ) -> List[Tuple[EdgeType, torch.Tensor, torch.Tensor]]:
        """Compute positive/negative scores grouped by edge type."""
        emb_dict = self.model(data)
        grouped_scores: List[Tuple[EdgeType, torch.Tensor, torch.Tensor]] = []

        for edge_type, edge_store in data.edge_items():
            src_type, _, dst_type = edge_type
            edge_index = edge_store.edge_index
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue

            # Subsample edges to control memory for expensive scorers
            if self.max_edges_per_type > 0 and num_edges > self.max_edges_per_type:
                perm = torch.randperm(num_edges, device=edge_index.device)[:self.max_edges_per_type]
                edge_index = edge_index[:, perm]
                edge_store_idx = edge_store.edge_type_idx[perm] if hasattr(edge_store, "edge_type_idx") else None
            else:
                edge_store_idx = edge_store.edge_type_idx if hasattr(edge_store, "edge_type_idx") else None

            head_emb = emb_dict[src_type][edge_index[0]]
            tail_emb = emb_dict[dst_type][edge_index[1]]

            if edge_store_idx is not None:
                rel_idx = edge_store_idx
            else:
                rel_idx = torch.zeros(
                    edge_index.size(1), dtype=torch.long, device=self.device
                )

            # For hybrid decoder models, pass relation indices directly so
            # the full HybridLinkScorer (DistMult+RotatE+Neural+Poincaré) is
            # used during training.  Otherwise use relation embeddings.
            _use_rel_idx = (
                hasattr(self.model, "_decoder_type")
                and self.model._decoder_type == "hybrid"
            )
            if _use_rel_idx:
                rel_for_score = rel_idx
            else:
                rel_for_score = self.model.get_relation_embedding(rel_idx)

            pos_scores = self.model.score(head_emb, rel_for_score, tail_emb)

            if self.negative_sampler is not None:
                neg_scores = self.negative_sampler(
                    head_emb, rel_for_score, tail_emb, emb_dict, edge_type
                )
            else:
                dst_pool_grouped = emb_dict[dst_type]
                # Restrict negative pool if applicable
                if self.neg_pool_restrictor is not None:
                    restricted_idx = self.neg_pool_restrictor(edge_type)
                    if restricted_idx is not None:
                        restricted_idx = restricted_idx.to(dst_pool_grouped.device)
                        dst_pool_grouped = dst_pool_grouped[restricted_idx]
                neg_scores = self._sample_negatives(
                    head_emb, rel_for_score, dst_pool_grouped, self.num_negatives,
                    src_pool=emb_dict[src_type], tail_emb=tail_emb,
                )

            grouped_scores.append((edge_type, pos_scores, neg_scores))

        return grouped_scores

    def _compute_loss_from_grouped_scores(
        self,
        grouped_scores: List[Tuple[EdgeType, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Aggregate relation losses with optional relation balancing."""
        if not grouped_scores:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if not self._edge_type_weights:
            pos_scores = torch.cat([x[1] for x in grouped_scores], dim=0)
            neg_scores = torch.cat([x[2] for x in grouped_scores], dim=0)
            return self.loss_fn(pos_scores, neg_scores)

        weighted_losses: List[torch.Tensor] = []
        total_weight = 0.0
        for edge_type, pos_scores, neg_scores in grouped_scores:
            weight = float(self._edge_type_weights.get(edge_type, 1.0))
            weighted_losses.append(self.loss_fn(pos_scores, neg_scores) * weight)
            total_weight += weight

        if total_weight <= 0:
            return torch.stack(weighted_losses).mean()
        return torch.stack(weighted_losses).sum() / total_weight

    def _compute_scores(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positive and negative triple scores.

        This is the default scoring pipeline.  It calls ``model.forward``
        to obtain embeddings, then scores all edge triples as positives and
        uses the negative sampler (if available) for negatives.
        """
        grouped_scores = self._compute_scores_grouped(data)
        if not grouped_scores:
            empty = torch.empty(0, device=self.device)
            return empty, empty

        pos_scores = torch.cat([x[1] for x in grouped_scores], dim=0)
        neg_scores = torch.cat([x[2] for x in grouped_scores], dim=0)
        return pos_scores, neg_scores

    def _simple_negative_scores(
        self, data: HeteroData, emb_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Generate negatives by randomly permuting tail indices."""
        neg_list: list[torch.Tensor] = []

        for edge_type, edge_store in data.edge_items():
            src_type, rel, dst_type = edge_type
            edge_index = edge_store.edge_index
            num_dst = emb_dict[dst_type].size(0)

            head_emb = emb_dict[src_type][edge_index[0]]
            # Random tails
            rand_idx = torch.randint(
                0, num_dst, (edge_index.size(1),), device=self.device
            )
            tail_emb = emb_dict[dst_type][rand_idx]

            if hasattr(edge_store, "edge_type_idx"):
                rel_idx = edge_store.edge_type_idx
            else:
                rel_idx = torch.zeros(
                    edge_index.size(1), dtype=torch.long, device=self.device
                )

            rel_emb = self.model.get_relation_embedding(rel_idx)
            neg_list.append(self.model.score(head_emb, rel_emb, tail_emb))

        return torch.cat(neg_list)

    def _training_step(
        self,
        grouped_scores: List[Tuple[EdgeType, torch.Tensor, torch.Tensor]],
    ) -> float:
        """One gradient update step."""
        self.optimizer.zero_grad()

        if self.mixed_precision:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                loss = self._compute_loss_from_grouped_scores(grouped_scores)

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.optimizer.step()
        else:
            loss = self._compute_loss_from_grouped_scores(grouped_scores)
            loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()

        return loss.item()

    def _accumulation_step(
        self,
        grouped_scores: List[Tuple[EdgeType, torch.Tensor, torch.Tensor]],
        accum_steps: int,
    ) -> float:
        """Compute loss and backward pass, scaling by 1/accum_steps.

        Unlike ``_training_step``, this does **not** zero gradients or
        call ``optimizer.step`` — the caller is responsible for that.
        """
        if self.mixed_precision:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                loss = self._compute_loss_from_grouped_scores(grouped_scores) / accum_steps
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            loss = self._compute_loss_from_grouped_scores(grouped_scores) / accum_steps
            loss.backward()

        return loss.item() * accum_steps  # return unscaled loss for logging

    def _optimizer_step(self) -> None:
        """Perform a single optimiser step (with optional grad clipping / scaler)."""
        if self._scaler is not None:
            if self.grad_clip_norm is not None:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()

    def _validate_loss(self) -> Dict[str, float]:
        """Quick validation using the same loss function."""
        self.model.eval()
        with torch.no_grad():
            grouped_scores = self._compute_scores_grouped(self.val_data)
            val_loss = self._compute_loss_from_grouped_scores(grouped_scores).item()
        return {"loss": val_loss}

    # ------------------------------------------------------------------
    # Callback dispatch
    # ------------------------------------------------------------------

    def _fire_on_train_begin(self) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(self)

    def _fire_on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end(self)

    def _fire_on_epoch_begin(self, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(self, epoch)

    def _fire_on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(self, epoch, train_loss, val_metrics)

    def _should_stop(self) -> bool:
        return any(cb.should_stop() for cb in self.callbacks)
