"""Training callbacks for early stopping, checkpointing, and logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# Base callback
# ======================================================================


class Callback:
    """Base class for trainer callbacks.

    Override any of the hook methods to customise training behaviour.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called once before training starts."""

    def on_train_end(self, trainer: Any) -> None:
        """Called once after training ends."""

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        """Called at the end of each epoch."""

    def should_stop(self) -> bool:
        """Return ``True`` to request early termination."""
        return False


# ======================================================================
# Early stopping
# ======================================================================


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    monitor : str
        Metric name to watch (default ``'mrr'``).
    patience : int
        Number of epochs with no improvement before stopping (default 30).
    min_delta : float
        Minimum change to qualify as an improvement (default 0.0).
    mode : str
        ``'max'`` if higher is better, ``'min'`` if lower is better.
    """

    def __init__(
        self,
        monitor: str = "mrr",
        patience: int = 30,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best: Optional[float] = None
        self._counter = 0
        self._stop = False

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        if val_metrics is None or self.monitor not in val_metrics:
            return

        current = val_metrics[self.monitor]

        if self._best is None:
            self._best = current
            return

        improved = (
            current > self._best + self.min_delta
            if self.mode == "max"
            else current < self._best - self.min_delta
        )

        if improved:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "EarlyStopping triggered: %s did not improve for %d epochs "
                    "(best=%.6f, current=%.6f)",
                    self.monitor,
                    self.patience,
                    self._best,
                    current,
                )
                self._stop = True

    def should_stop(self) -> bool:
        return self._stop


# ======================================================================
# Model checkpoint
# ======================================================================


class ModelCheckpoint(Callback):
    """Save model checkpoints when a monitored metric improves.

    Parameters
    ----------
    dirpath : str or Path
        Directory to save checkpoints.
    monitor : str
        Metric name to watch (default ``'mrr'``).
    mode : str
        ``'max'`` or ``'min'`` (default ``'max'``).
    save_last : bool
        Always keep a ``last.pt`` checkpoint (default ``True``).
    """

    def __init__(
        self,
        dirpath: str | Path,
        monitor: str = "mrr",
        mode: str = "max",
        save_last: bool = True,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last

        self._best: Optional[float] = None

    def on_train_begin(self, trainer: Any) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }

        # Always save last
        if self.save_last:
            torch.save(state, self.dirpath / "last.pt")

        if val_metrics is None or self.monitor not in val_metrics:
            return

        current = val_metrics[self.monitor]
        improved = (
            self._best is None
            or (self.mode == "max" and current > self._best)
            or (self.mode == "min" and current < self._best)
        )

        if improved:
            self._best = current
            torch.save(state, self.dirpath / "best.pt")
            logger.info(
                "Checkpoint saved: %s improved to %.6f (epoch %d)",
                self.monitor,
                current,
                epoch,
            )


# ======================================================================
# Metrics logger
# ======================================================================


class MetricsLogger(Callback):
    """Log training metrics to a JSON-lines file and optionally to W&B.

    Parameters
    ----------
    log_file : str or Path or None
        Path to a ``.jsonl`` log file.  If ``None``, only console logging.
    use_wandb : bool
        If ``True``, also log to Weights & Biases (must be initialised
        externally).
    use_tensorboard : bool
        If ``True``, also log to TensorBoard.
    tensorboard_dir : str or Path or None
        TensorBoard log directory (required if ``use_tensorboard=True``).
    """

    def __init__(
        self,
        log_file: Optional[str | Path] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        tensorboard_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.log_file = Path(log_file) if log_file is not None else None
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self._tb_writer = None

        if use_tensorboard:
            if tensorboard_dir is None:
                raise ValueError("tensorboard_dir is required when use_tensorboard=True")
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        record: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if val_metrics is not None:
            record.update(val_metrics)

        # Console
        parts = [f"epoch={epoch}", f"loss={train_loss:.4f}"]
        if val_metrics:
            parts.extend(f"{k}={v:.4f}" for k, v in val_metrics.items())
        logger.info("  ".join(parts))

        # JSON-lines
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as fh:
                fh.write(json.dumps(record) + "\n")

        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb

                wandb.log(record, step=epoch)
            except ImportError:
                logger.warning("wandb not installed; skipping W&B logging")

        # TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss", train_loss, epoch)
            if val_metrics:
                for k, v in val_metrics.items():
                    self._tb_writer.add_scalar(f"val/{k}", v, epoch)

    def on_train_end(self, trainer: Any) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
