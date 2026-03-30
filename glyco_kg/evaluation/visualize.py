"""Embedding visualisation utilities for glycoMusubi.

Provides t-SNE and UMAP dimensionality reduction with publication-quality
Matplotlib figures suitable for Nature / Science submissions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a tensor or array to a numpy ndarray on CPU."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


class EmbeddingVisualizer:
    """Visualise entity and relation embeddings via dimensionality reduction.

    Parameters
    ----------
    figsize : tuple of float
        Default figure size ``(width, height)`` in inches.
    dpi : int
        Resolution for saved figures.
    style : str
        Matplotlib style to apply (e.g. ``"seaborn-v0_8-whitegrid"``).
    random_state : int
        Seed for reproducible reductions.
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 300,
        style: str = "seaborn-v0_8-whitegrid",
        random_state: int = 42,
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.random_state = random_state

    # -----------------------------------------------------------------
    # Entity embeddings
    # -----------------------------------------------------------------

    def plot_embeddings(
        self,
        embeddings_dict: Dict[str, ArrayLike],
        method: str = "tsne",
        output_path: Optional[Union[str, Path]] = None,
        *,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        title: str = "Entity Embeddings",
        point_size: float = 8.0,
        alpha: float = 0.7,
        legend_fontsize: int = 10,
        cmap: Optional[str] = None,
        max_points_per_type: Optional[int] = None,
    ) -> "matplotlib.figure.Figure":
        """Create a 2-D scatter plot of entity embeddings coloured by type.

        Parameters
        ----------
        embeddings_dict : dict[str, array-like]
            Mapping from entity-type name (e.g. ``"glycan"``) to an
            embedding matrix of shape ``[N_type, dim]``.
        method : ``"tsne"`` or ``"umap"``
            Dimensionality-reduction algorithm.
        output_path : str or Path or None
            If given, the figure is saved to this path.
        perplexity : float
            t-SNE perplexity (ignored for UMAP).
        n_neighbors : int
            UMAP ``n_neighbors`` (ignored for t-SNE).
        min_dist : float
            UMAP ``min_dist`` (ignored for t-SNE).
        title : str
            Plot title.
        point_size : float
            Marker size in scatter plot.
        alpha : float
            Marker transparency.
        legend_fontsize : int
            Font size for legend labels.
        cmap : str or None
            Matplotlib colourmap name.  If ``None``, a categorical palette
            is generated automatically.
        max_points_per_type : int or None
            If set, randomly sub-sample each type to at most this many
            points (useful for very large KGs).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        import matplotlib.pyplot as plt

        all_embeddings, labels, type_names = self._stack_embeddings(
            embeddings_dict, max_points_per_type
        )
        if all_embeddings.shape[0] == 0:
            logger.warning("No embeddings to plot.")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title(title)
            return fig

        coords = self._reduce(all_embeddings, method, perplexity, n_neighbors, min_dist)

        colours = self._get_palette(len(type_names), cmap)

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            for idx, tname in enumerate(type_names):
                mask = labels == idx
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=point_size,
                    alpha=alpha,
                    c=[colours[idx]],
                    label=f"{tname} (n={mask.sum()})",
                    edgecolors="none",
                )
            ax.set_title(title, fontsize=14, fontweight="bold")
            method_label = method.upper()
            ax.set_xlabel(f"{method_label}-1", fontsize=12)
            ax.set_ylabel(f"{method_label}-2", fontsize=12)
            ax.legend(
                fontsize=legend_fontsize,
                markerscale=2.0,
                frameon=True,
                loc="best",
            )
            fig.tight_layout()

        if output_path is not None:
            fig.savefig(str(output_path), dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved embedding plot to %s", output_path)

        return fig

    # -----------------------------------------------------------------
    # Relation embeddings
    # -----------------------------------------------------------------

    def plot_relation_embeddings(
        self,
        relation_embeddings: ArrayLike,
        relation_names: Optional[Sequence[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        *,
        method: str = "tsne",
        perplexity: float = 5.0,
        n_neighbors: int = 5,
        min_dist: float = 0.3,
        title: str = "Relation Embeddings",
        point_size: float = 80.0,
        annotate: bool = True,
    ) -> "matplotlib.figure.Figure":
        """Visualise relation embeddings in 2-D with text annotations.

        Parameters
        ----------
        relation_embeddings : array-like
            Shape ``[num_relations, dim]``.
        relation_names : sequence of str or None
            Human-readable names for each relation.  If ``None``, uses
            integer indices.
        output_path : str or Path or None
            If given, the figure is saved to this path.
        method : ``"tsne"`` or ``"umap"``
            Dimensionality-reduction algorithm.
        perplexity : float
            t-SNE perplexity.
        n_neighbors : int
            UMAP n_neighbors.
        min_dist : float
            UMAP min_dist.
        title : str
            Plot title.
        point_size : float
            Marker size.
        annotate : bool
            Whether to annotate each point with its relation name.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        emb = _to_numpy(relation_embeddings)
        n_rel = emb.shape[0]

        if relation_names is None:
            relation_names = [str(i) for i in range(n_rel)]

        if n_rel < 2:
            logger.warning(
                "Fewer than 2 relations — skipping dimensionality reduction."
            )
            coords = emb[:, :2] if emb.shape[1] >= 2 else np.zeros((n_rel, 2))
        else:
            # Adjust perplexity for small relation sets.
            safe_perplexity = min(perplexity, max(1.0, (n_rel - 1) / 3.0))
            safe_n_neighbors = min(n_neighbors, n_rel - 1)
            coords = self._reduce(emb, method, safe_perplexity, safe_n_neighbors, min_dist)

        colours = self._get_palette(n_rel)

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            for i in range(n_rel):
                ax.scatter(
                    coords[i, 0],
                    coords[i, 1],
                    s=point_size,
                    c=[colours[i]],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                )
                if annotate:
                    ax.annotate(
                        relation_names[i],
                        (coords[i, 0], coords[i, 1]),
                        textcoords="offset points",
                        xytext=(8, 4),
                        fontsize=10,
                        fontweight="bold",
                    )
            ax.set_title(title, fontsize=14, fontweight="bold")
            method_label = method.upper()
            ax.set_xlabel(f"{method_label}-1", fontsize=12)
            ax.set_ylabel(f"{method_label}-2", fontsize=12)
            fig.tight_layout()

        if output_path is not None:
            fig.savefig(str(output_path), dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved relation plot to %s", output_path)

        return fig

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _stack_embeddings(
        self,
        embeddings_dict: Dict[str, ArrayLike],
        max_points: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Stack per-type embeddings into a single array with label vector."""
        rng = np.random.RandomState(self.random_state)
        arrays: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        type_names: List[str] = []

        for idx, (tname, emb) in enumerate(sorted(embeddings_dict.items())):
            arr = _to_numpy(emb)
            if arr.ndim != 2:
                raise ValueError(
                    f"Embeddings for '{tname}' must be 2-D [N, dim], "
                    f"got shape {arr.shape}"
                )
            if max_points is not None and arr.shape[0] > max_points:
                indices = rng.choice(arr.shape[0], size=max_points, replace=False)
                arr = arr[indices]
            arrays.append(arr)
            labels.append(np.full(arr.shape[0], idx, dtype=np.int32))
            type_names.append(tname)

        if not arrays:
            return np.empty((0, 0)), np.empty(0, dtype=np.int32), []

        return np.vstack(arrays), np.concatenate(labels), type_names

    def _reduce(
        self,
        embeddings: np.ndarray,
        method: str,
        perplexity: float,
        n_neighbors: int,
        min_dist: float,
    ) -> np.ndarray:
        """Apply dimensionality reduction to 2-D."""
        n_samples = embeddings.shape[0]
        if n_samples <= 2 or embeddings.shape[1] <= 2:
            # No reduction needed / possible.
            if embeddings.shape[1] >= 2:
                return embeddings[:, :2].copy()
            return np.zeros((n_samples, 2))

        method_lower = method.lower()
        if method_lower == "tsne":
            from sklearn.manifold import TSNE

            safe_perplexity = min(perplexity, max(1.0, (n_samples - 1) / 3.0))
            reducer = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                random_state=self.random_state,
                init="pca",
                learning_rate="auto",
            )
            return reducer.fit_transform(embeddings)

        if method_lower == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP requires the 'umap-learn' package. "
                    "Install it with:  pip install umap-learn"
                ) from exc
            safe_n_neighbors = min(n_neighbors, n_samples - 1)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=safe_n_neighbors,
                min_dist=min_dist,
                random_state=self.random_state,
            )
            return reducer.fit_transform(embeddings)

        raise ValueError(
            f"Unknown reduction method '{method}'. Use 'tsne' or 'umap'."
        )

    @staticmethod
    def _get_palette(
        n: int, cmap: Optional[str] = None
    ) -> List[Tuple[float, float, float, float]]:
        """Generate *n* visually distinct RGBA colours."""
        import matplotlib.pyplot as plt

        if cmap is not None:
            cm = plt.get_cmap(cmap)
            return [cm(i / max(n - 1, 1)) for i in range(n)]

        # Use tab10 / tab20 for small n, otherwise sample from HSV.
        if n <= 10:
            cm = plt.get_cmap("tab10")
        elif n <= 20:
            cm = plt.get_cmap("tab20")
        else:
            cm = plt.get_cmap("hsv")
        return [cm(i / max(n - 1, 1)) for i in range(n)]
