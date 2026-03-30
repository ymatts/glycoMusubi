"""Poincare ball distance model for hyperbolic knowledge graph embeddings.

Implements the Poincare ball model B^d_c with configurable curvature *c*,
providing Mobius addition, exponential/logarithmic maps, and a differentiable
distance function suitable for scoring hierarchical relations in glycoMusubi.

References
----------
- Section 4.2.4 of ``docs/design/algorithm_design.md``
- Appendix B of ``docs/design/algorithm_design.md``
- Nickel & Kiela, "Poincare Embeddings for Learning Hierarchical Representations", NeurIPS 2017
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoincareDistance(nn.Module):
    """Poincare ball distance scorer for hyperbolic link prediction.

    Operates in the Poincare ball model B^d_c = {x in R^d : c * ||x||^2 < 1}
    with curvature parameter *c*.

    The scoring function is:

    ``S_hyp(h, r, t) = -d_c(exp_0(e_h + r), exp_0(e_t))``

    where ``d_c`` is the Poincare distance and ``exp_0`` is the exponential map
    from the tangent space at the origin into the ball.

    Parameters
    ----------
    curvature : float
        Curvature of the Poincare ball (default 1.0).  Must be positive.
    eps : float
        Small constant for numerical stability (default 1e-5).
    max_norm : float
        Maximum norm for points inside the ball; norms are clamped to
        ``1 - eps`` / sqrt(c) to stay within the open ball.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        eps: float = 1e-5,
        max_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if curvature <= 0:
            raise ValueError(f"Curvature must be positive, got {curvature}")
        self.c = curvature
        self.eps = eps
        # Points must satisfy c * ||x||^2 < 1, so ||x|| < 1/sqrt(c).
        self.max_norm = (1.0 - eps) / (curvature ** 0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Project points back into the open Poincare ball."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        clamped = x / norm * norm.clamp(max=self.max_norm)
        # Only project if norm exceeds max_norm.
        mask = (norm > self.max_norm).float()
        return x * (1 - mask) + clamped * mask

    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor lambda_x^c = 2 / (1 - c * ||x||^2)."""
        x_sqnorm = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0 / self.c - self.eps)
        return 2.0 / (1.0 - self.c * x_sqnorm).clamp(min=self.eps)

    # ------------------------------------------------------------------
    # Mobius addition
    # ------------------------------------------------------------------

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition in the Poincare ball.

        ``x oplus_c y = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y)
                        / (1 + 2c<x,y> + c^2 ||x||^2 ||y||^2)``

        Parameters
        ----------
        x : torch.Tensor
            ``[..., d]``
        y : torch.Tensor
            ``[..., d]``

        Returns
        -------
        torch.Tensor
            ``[..., d]``
        """
        x = self._clamp_norm(x)
        y = self._clamp_norm(y)

        c = self.c
        x_sqnorm = (x * x).sum(dim=-1, keepdim=True)
        y_sqnorm = (y * y).sum(dim=-1, keepdim=True)
        xy_dot = (x * y).sum(dim=-1, keepdim=True)

        numerator = (1 + 2 * c * xy_dot + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
        denominator = 1 + 2 * c * xy_dot + c ** 2 * x_sqnorm * y_sqnorm
        denominator = denominator.clamp(min=self.eps)

        result = numerator / denominator
        return self._clamp_norm(result)

    # ------------------------------------------------------------------
    # Exponential map
    # ------------------------------------------------------------------

    def exp_map(
        self, v: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Exponential map from tangent space at *x* to the Poincare ball.

        If *x* is ``None``, uses the origin (exp_0).

        ``exp_x^c(v) = x oplus_c (tanh(sqrt(c) * lambda_x * ||v|| / 2) * v / (sqrt(c) * ||v||))``

        Parameters
        ----------
        v : torch.Tensor
            Tangent vector ``[..., d]``.
        x : torch.Tensor or None
            Base point ``[..., d]``.  If ``None``, the origin is used.

        Returns
        -------
        torch.Tensor
            Point on the Poincare ball ``[..., d]``.
        """
        sqrt_c = self.c ** 0.5
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        if x is None:
            # exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
            second_term = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
            return self._clamp_norm(second_term)

        x = self._clamp_norm(x)
        lam = self._lambda_x(x)
        second_term = (
            torch.tanh(sqrt_c * lam * v_norm / 2.0) * v / (sqrt_c * v_norm)
        )
        return self.mobius_add(x, self._clamp_norm(second_term))

    # ------------------------------------------------------------------
    # Logarithmic map
    # ------------------------------------------------------------------

    def log_map(
        self, y: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Logarithmic map from the Poincare ball to the tangent space at *x*.

        If *x* is ``None``, uses the origin (log_0).

        ``log_x^c(y) = (2 / (sqrt(c) * lambda_x)) * arctanh(sqrt(c) * ||(-x) oplus_c y||) * ((-x) oplus_c y) / ||(-x) oplus_c y||``

        Parameters
        ----------
        y : torch.Tensor
            Point on the ball ``[..., d]``.
        x : torch.Tensor or None
            Base point ``[..., d]``.  If ``None``, the origin is used.

        Returns
        -------
        torch.Tensor
            Tangent vector ``[..., d]``.
        """
        sqrt_c = self.c ** 0.5
        y = self._clamp_norm(y)

        if x is None:
            # log_0(y) = arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
            y_norm = y.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            return torch.atanh(
                (sqrt_c * y_norm).clamp(max=1.0 - self.eps)
            ) * y / (sqrt_c * y_norm)

        x = self._clamp_norm(x)
        neg_x = -x
        add_result = self.mobius_add(neg_x, y)
        add_norm = add_result.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        lam = self._lambda_x(x)
        return (
            (2.0 / (sqrt_c * lam))
            * torch.atanh((sqrt_c * add_norm).clamp(max=1.0 - self.eps))
            * add_result
            / add_norm
        )

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Poincare distance between two points.

        ``d_c(x, y) = (2 / sqrt(c)) * arctanh(sqrt(c) * ||(-x) oplus_c y||)``

        Parameters
        ----------
        x : torch.Tensor
            ``[..., d]``
        y : torch.Tensor
            ``[..., d]``

        Returns
        -------
        torch.Tensor
            Pairwise distances ``[...]``.
        """
        sqrt_c = self.c ** 0.5
        neg_x = -self._clamp_norm(x)
        y = self._clamp_norm(y)
        add_result = self.mobius_add(neg_x, y)
        add_norm = add_result.norm(dim=-1).clamp(min=self.eps)
        return (2.0 / sqrt_c) * torch.atanh(
            (sqrt_c * add_norm).clamp(max=1.0 - self.eps)
        )

    # ------------------------------------------------------------------
    # Forward (scoring)
    # ------------------------------------------------------------------

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of triples using hyperbolic distance.

        ``S_hyp(h, r, t) = -d_c(exp_0(e_h + r), exp_0(e_t))``

        Head and relation are added in tangent space at origin before mapping
        into the Poincare ball.

        Parameters
        ----------
        head : torch.Tensor
            Head entity embeddings ``[batch, dim]``.
        relation : torch.Tensor
            Relation embeddings ``[batch, dim]``.
        tail : torch.Tensor
            Tail entity embeddings ``[batch, dim]``.

        Returns
        -------
        torch.Tensor
            Scores ``[batch]`` (higher is more plausible).
        """
        # Translate head by relation in tangent space, then project into ball.
        hr = self.exp_map(head + relation)  # exp_0(e_h + r)
        t = self.exp_map(tail)              # exp_0(e_t)
        return -self.distance(hr, t)
