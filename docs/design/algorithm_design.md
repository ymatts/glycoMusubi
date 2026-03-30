# glycoMusubi Embedding Algorithm Design

## Author: Implementation Design Specialist 1 (Algorithm Design)
## Date: 2026-02-13
## Status: COMPLETE

---

## 1. Problem Characterization

### 1.1 glycoMusubi Graph Properties

Based on analysis of the existing codebase (`scripts/build_kg.py`, `schemas/`), the glycoMusubi possesses the following structural characteristics:

| Property | Value | Implication |
|----------|-------|-------------|
| **Node types** | 9+ (glycan, protein, enzyme, disease, variant, compound, motif, reaction, pathway, site) | Heterogeneous graph; type-aware embeddings required |
| **Relation types** | 12+ (has_glycan, inhibits, associated_with_disease, has_variant, produced_by, consumed_by, child_of, has_motif, catalyzed_by, has_product, has_site, ptm_crosstalk) | Multi-relational; composition operators needed |
| **Scale** | 10^5-10^6 nodes, 10^6-10^7 edges | Moderate; GNN and shallow methods both feasible |
| **Glycan internal structure** | Tree/DAG (WURCS-encoded branching oligosaccharides) | Requires specialized structural encoder |
| **Hierarchy** | Glycan subsumption (child_of), motif containment, pathway membership | Hyperbolic geometry advantageous |
| **Incompleteness** | Open World Assumption; sparse protein-disease, glycan-protein links | Robust negative sampling; calibrated scoring |
| **Directionality** | Mixed: some symmetric (ptm_crosstalk), most asymmetric | Must model symmetry/antisymmetry patterns |

### 1.2 Relation Pattern Analysis

The glycoMusubi relation types exhibit the following algebraic patterns that the embedding model must faithfully capture:

| Pattern | Relations in glycoMusubi | Mathematical Requirement |
|---------|----------------------|--------------------------|
| **Antisymmetry** | inhibits, has_glycan, has_variant, produced_by, catalyzed_by | r(h,t) => not r(t,h) |
| **Symmetry** | ptm_crosstalk | r(h,t) <=> r(t,h) |
| **Composition** | protein -[has_site]-> site -[ptm_crosstalk]-> site -[has_site^-1]-> protein | r1 . r2 = r3 |
| **Hierarchy** | child_of (glycan subsumption), has_motif | Partial order, transitivity |
| **1-to-N** | protein -[has_glycan]-> {glycan1, glycan2, ...} | One-to-many mapping |
| **N-to-1** | {compound1, compound2, ...} -[inhibits]-> enzyme | Many-to-one mapping |
| **N-to-N** | protein <-[associated_with_disease]-> disease | Many-to-many mapping |

---

## 2. Algorithm Comparison

### 2.1 Comprehensive Comparison Table

| Method | Space | Scoring Function | Relation Patterns (Sym/Anti/Comp/Inv/Hier) | Time Complexity (per triple) | Memory | Inductive? | Interpretable? |
|--------|-------|-----------------|---------------------------------------------|------------------------------|--------|-----------|----------------|
| **TransE** | R^d | -\|\|h+r-t\|\| | No/Yes/Yes/Yes/No | O(d) | O(n_e*d + n_r*d) | No | Low |
| **RotatE** | C^d | -\|\|h.r-t\|\| | Yes/Yes/Yes/Yes/No | O(d) | O(n_e*d + n_r*d) | No | Low |
| **PairRE** | R^d | -\|\|h.r_H - t.r_T\|\| | Yes/Yes/Yes/Yes/Partial | O(d) | O(n_e*d + 2*n_r*d) | No | Low |
| **HousE** | R^d | Householder(h,r)-t | Yes/Yes/Yes/Yes/Yes | O(k*d) | O(n_e*d + n_r*k*d) | No | Low |
| **DistMult** | R^d | h^T diag(r) t | Yes/No/No/No/No | O(d) | O(n_e*d + n_r*d) | No | Low |
| **ComplEx** | C^d | Re(h^T diag(r) conj(t)) | Yes/Yes/No/Yes/No | O(d) | O(n_e*d + n_r*d) | No | Low |
| **R-GCN** | R^d | GNN + DistMult/TransE | Depends on decoder | O(|E|*d^2/n_r) | O(n_r*d^2) | Semi | Medium |
| **CompGCN** | R^d | GNN + composition ops | Yes/Yes/Yes/Yes/Partial | O(|E|*d) | O((n_e+n_r)*d) | Semi | Medium |
| **NBFNet** | R^d | Bellman-Ford GNN | All (path-based) | O(|E|*d*T) | O(n_e*d) | Yes | **High** |
| **BioPathNet** | R^d | NBFNet + background graph | All (path-based) | O(|E|*d*T) | O(n_e*d) | Yes | **High** |
| **ULTRA** | R^d | Conditional NBFNet | All (zero-shot transfer) | O(|E|*d*T) | O(d) (no entity params) | **Yes** | High |
| **HypE (Poincare)** | B^d_c | Mobius addition | Yes/Yes/Partial/Yes/**Yes** | O(d) | O(n_e*d + n_r*d) | No | Medium |
| **LorentzKG** | H^d | Lorentz rotation + boost | Yes/Yes/Yes/Yes/**Yes** | O(d^2) | O(n_e*d + n_r*d^2) | No | Medium |
| **Fully Hyp. RotatE** | H^d | Lorentz rotation | Yes/Yes/Yes/Yes/**Yes** | O(d) | O(n_e*d + n_r*d) | No | Medium |

Where: n_e = number of entities, n_r = number of relations, d = embedding dimension, |E| = number of edges, T = number of GNN layers/iterations, k = Householder reflection count.

### 2.2 Suitability Analysis for glycoMusubi

**Critical requirements** (must satisfy):
1. Handle 9+ node types and 12+ relation types
2. Model both symmetric and antisymmetric relations
3. Capture hierarchical structure (glycan subsumption, motif containment)
4. Scale to 10^6 nodes / 10^7 edges
5. Handle incompleteness (Open World Assumption)

**Highly desirable** (Nature/Science impact):
6. Encode glycan branching structure (tree/DAG) as inductive bias
7. Provide interpretable predictions (path-based reasoning)
8. Support inductive inference (new glycans/proteins not seen during training)
9. Novel contribution over existing biomedical KG methods

---

## 3. Recommended Algorithms (Top 3)

### 3.1 Recommendation 1: GlycoPathNet (Novel Hybrid -- Primary Proposal)

**Architecture**: Glycan Structure Encoder (GlycanEncoder) + Path-based KG Reasoner (adapted BioPathNet/NBFNet)

**Rationale**: This is a novel architecture that combines domain-specific inductive bias for glycan branching structures with state-of-the-art path-based reasoning. No existing method jointly addresses glycan structural encoding and KG-level link prediction.

**Key Innovation**: The glycan internal structure (tree/DAG) is encoded via a hierarchical message-passing GNN that produces glycan node embeddings aware of monosaccharide composition, linkage types, and branching topology. These structure-aware embeddings are then injected into a path-based KG reasoning framework.

**Why this is the best choice**:
- BioPathNet (Nature Biomedical Engineering, 2025) demonstrated SOTA on biomedical KGs
- SweetNet/GIFFLAR demonstrated that GNN-based glycan encoding outperforms sequence models
- GlycanAA (ICML 2025) showed hierarchical message passing captures atom-to-monosaccharide-to-glycan scales
- No existing work combines these two lines of research
- Interpretable: paths through the KG can be traced and biologically validated

### 3.2 Recommendation 2: Hyperbolic CompGCN with Glycan Structure Encoding

**Architecture**: Glycan Structure Encoder + CompGCN in mixed curvature space (Euclidean for non-hierarchical, hyperbolic for hierarchical relations)

**Rationale**: CompGCN (ICLR 2020) naturally handles multi-relational graphs with composition operations. Extending it to hyperbolic space captures the hierarchical nature of glycan subsumption and biological taxonomies. The glycan structure encoder provides domain-specific inductive bias.

**Why this is a strong alternative**:
- CompGCN scales linearly with number of edges
- Hyperbolic spaces provide exponentially more room for hierarchical data
- LorentzKG and Fully Hyperbolic RotatE (2024) show hyperbolic KGE is mature
- Mixed-curvature allows different geometry for different relation types
- More parameter-efficient than path-based methods

### 3.3 Recommendation 3: ULTRA-Glyco (Zero-Shot Foundation Model Approach)

**Architecture**: Fine-tuned ULTRA with glycan-specific relation graph conditioning

**Rationale**: ULTRA (ICLR 2024) represents the frontier of KG foundation models. A glycan-specific fine-tuning of ULTRA would enable zero-shot generalization to unseen glycan structures and relations, which is critical for the rapidly expanding glycomics field.

**Why this is valuable**:
- Pre-trained on 57 KGs; transferable structural knowledge
- Zero-shot inference on new entities/relations without retraining
- As glycomics databases grow, new glycans appear constantly
- Eliminates need for entity-specific parameters
- However: does not exploit glycan internal structure (weakness)

---

## 4. Novel Algorithm: GlycoPathNet

### 4.1 Overview

GlycoPathNet is a two-stage architecture:

```
Stage 1: GlycanEncoder -- encodes internal glycan structure into d-dimensional embedding
Stage 2: PathReasoner  -- performs path-based link prediction on the heterogeneous KG
```

The key novelty is that **glycan node embeddings are not learned as free parameters** but are **computed from their molecular structure**, enabling:
- Inductive generalization to unseen glycans
- Structure-activity relationship discovery
- Interpretable substructure-level attention

### 4.2 Mathematical Definition

#### 4.2.1 GlycanEncoder: Hierarchical Tree-GNN

Given a glycan g represented as a rooted tree T_g = (V_g, E_g) where:
- V_g = {v_1, ..., v_m} are monosaccharide nodes
- E_g = {e_1, ..., e_{m-1}} are glycosidic linkage edges
- Each v_i has features x_i in R^{d_mono} (monosaccharide type, ring form, anomeric config)
- Each e_j has features l_j in R^{d_link} (linkage position, e.g., alpha-1,3)

**Layer-wise update** (L layers, bottom-up then top-down):

Bottom-up pass (leaves to root):
```
h_v^(0) = W_node * x_v + b_node                           -- initial node embedding
m_v^(l) = AGG({MSG(h_u^(l-1), h_v^(l-1), l_{uv}) : u in Children(v)})
h_v^(l) = UPDATE(h_v^(l-1), m_v^(l))
```

Where:
```
MSG(h_u, h_v, l_{uv}) = sigma(W_msg * [h_u || h_v || l_{uv}])
AGG = attention-weighted sum with scores alpha_{uv} = softmax(a^T * [h_u || h_v])
UPDATE(h, m) = GRU(h, m)  -- Gated Recurrent Unit for stable gradient flow
```

Top-down pass (root to leaves):
```
h_v^(td,l) = UPDATE_td(h_v^(L), m_v^(td,l))
m_v^(td,l) = MSG_td(h_{parent(v)}^(td,l-1), h_v^(L), l_{parent(v),v})
```

**Glycan-level readout**:
```
z_g = READOUT({h_v^(final) : v in V_g})
    = W_read * [mean_pool(H) || max_pool(H) || h_root^(final)] + b_read
```

where H = {h_v^(final)} is the set of all final node embeddings, and h_root is the root monosaccharide embedding.

**Output**: z_g in R^d is the structure-aware glycan embedding.

#### 4.2.2 Entity Embedding Function

For the full KG, entity embeddings are computed as:
```
e_v = {
  GlycanEncoder(T_v)                       if type(v) = glycan
  W_{type(v)} * x_v + b_{type(v)}          if type(v) in {protein, enzyme, ...} and features available
  embed_lookup(v)                           if no structural features available
}
```

This is a **hybrid** approach: glycan nodes get structure-derived embeddings; other nodes use learnable embeddings or feature projections (e.g., from protein sequence embeddings via ESM-2).

#### 4.2.3 PathReasoner: Generalized Bellman-Ford on Heterogeneous Graph

Following NBFNet/BioPathNet, we define the path-based score for a query (h, r, ?):

**Initialization** (boundary condition):
```
h_v^(0) = INDICATOR(v, h) = {
  e_h    if v = h
  0      otherwise
}
```

**Iterative message passing** (T iterations):
```
h_v^(t) = AGG_KG({MSG_KG(h_u^(t-1), r_{uv}) : (u, r_{uv}, v) in E_KG})

MSG_KG(h_u, r_{uv}) = W_r * h_u + b_r        -- relation-specific linear transform
                     = sigma(MLP([h_u || e_{r_{uv}}]))  -- or MLP-based

AGG_KG = sum or PNA (Principal Neighbourhood Aggregation)
```

**Final scoring**:
```
score(h, r, t) = MLP_score([h_t^(T) || e_r])
```

This computes the generalized sum of all path representations from h to t, weighted by learned relation-specific message functions.

#### 4.2.4 Composite Scoring Function (Novel)

We propose a **multi-view scoring function** that combines path-based reasoning with direct structural similarity:

```
S(h, r, t) = alpha * S_path(h, r, t) + beta * S_struct(h, r, t) + gamma * S_hyp(h, r, t)

where:
  S_path(h, r, t)   = MLP([h_t^(T) || e_r])              -- path-based score
  S_struct(h, r, t)  = cos(z_h, W_r^struct * z_t)         -- structural similarity (for glycan pairs)
  S_hyp(h, r, t)     = -d_H(exp_o(e_h + r), exp_o(e_t))  -- hyperbolic distance for hierarchical relations
```

- alpha, beta, gamma are learned relation-type-dependent weights
- S_struct is only active when both h, t are glycans (e.g., for child_of, has_motif relations)
- S_hyp uses the Poincare ball distance d_H for hierarchical relations
- exp_o is the exponential map from tangent space to Poincare ball

#### 4.2.5 Loss Function

Binary cross-entropy with self-adversarial negative sampling:

```
L = - (1/|T|) * sum_{(h,r,t) in T} [
    log sigma(S(h,r,t))
    + (1/K) * sum_{i=1}^{K} p(h_i',r,t_i') * log sigma(-S(h_i',r,t_i'))
]

p(h_i',r,t_i') = softmax(alpha_neg * S(h_i',r,t_i'))  -- self-adversarial weight
```

Where K is the number of negative samples, and alpha_neg is the adversarial temperature.

**Additional regularization terms**:
```
L_total = L + lambda_1 * L_struct + lambda_2 * L_hyp + lambda_3 * ||theta||^2

L_struct = contrastive loss on glycan structural embeddings (glycans with shared motifs should be close)
L_hyp = Riemannian gradient penalty for hyperbolic embeddings
```

### 4.3 Theoretical Analysis: Expressiveness

**Theorem 1 (Relation Pattern Completeness)**:
GlycoPathNet can model all fundamental relation patterns:

| Pattern | Mechanism |
|---------|-----------|
| **Symmetry** | Path-based: paths h->t and t->h learned independently; S_path can output equal scores |
| **Antisymmetry** | Path-based: asymmetric message passing with directed edges |
| **Composition** | Path-based: T-iteration message passing captures paths of length T; composition r1.r2=r3 is a 2-hop path |
| **Inversion** | Path-based: inverse edges explicitly added to message passing |
| **Hierarchy** | S_hyp: Poincare distance naturally encodes tree-like partial orders |
| **1-to-N / N-to-1** | PairRE-like paired relation vectors in S_struct allow non-injective mappings |

**Theorem 2 (Structural Induction)**:
For any unseen glycan g' with tree structure T_{g'}, GlycoPathNet produces a valid embedding z_{g'} = GlycanEncoder(T_{g'}) without retraining, provided the monosaccharide vocabulary is covered.

*Proof sketch*: GlycanEncoder is a permutation-equivariant message-passing GNN over the glycan tree. By the universality of GNN + global readout (Xu et al., 2019), GlycanEncoder can distinguish non-isomorphic glycan trees up to the 1-WL test. Since glycan trees are labeled (monosaccharide types + linkage types), GlycanEncoder can distinguish any two glycans with different compositions or topologies.

**Theorem 3 (Path Expressiveness)**:
With T iterations of Bellman-Ford message passing, GlycoPathNet considers all paths of length <= T between query entity h and candidate entity t. The generalized Bellman-Ford framework subsumes:
- TransE (single-hop, additive)
- RotatE (single-hop, multiplicative)
- Any fixed-length path scoring

*Proof*: Follows directly from Zhu et al. (2021, NeurIPS) -- NBFNet generalizes all pair-specific path-based methods.

### 4.4 Computational Complexity

| Component | Time (training, per epoch) | Memory |
|-----------|---------------------------|--------|
| GlycanEncoder (per glycan) | O(L * m * d^2) where m = avg monosaccharides per glycan (~10-20) | O(n_glycan * d) |
| PathReasoner (per batch) | O(T * \|E\| * d) | O(n_e * d) |
| Scoring | O(B * d) where B = batch size | O(B * d) |
| **Total per epoch** | O(T * \|E\| * d + n_glycan * L * m * d^2) | O((n_e + n_r) * d) |

For glycoMusubi scale (n_e ~ 10^6, |E| ~ 10^7, d = 256, T = 6):
- Training time per epoch: ~10^7 * 6 * 256 ~ 1.5 * 10^10 FLOPs (feasible on single GPU)
- Memory: ~10^6 * 256 * 4 bytes ~ 1 GB for entity embeddings

### 4.5 Comparison with Existing Biomedical KG Methods

| Method | Glycan Structure Aware? | Hierarchical? | Inductive? | Path Interpretable? | Novel? |
|--------|------------------------|---------------|-----------|---------------------|--------|
| BioPathNet | No | No | Yes | Yes | No (published 2025) |
| DRKG+RotatE | No | No | No | No | No |
| HEM (hyperbolic bio KGE) | No | Yes | No | No | No (published 2024) |
| SweetNet | Yes (glycan only) | No | Yes (glycan) | No | No (published 2021) |
| GIFFLAR | Yes (glycan only) | No | Yes (glycan) | No | No (published 2024) |
| GlycanAA | Yes (all-atom) | Partial | Yes (glycan) | No | No (published 2025) |
| **GlycoPathNet (ours)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

GlycoPathNet is the **first** method to jointly:
1. Encode glycan internal branching structure via specialized Tree-GNN
2. Perform path-based reasoning across a heterogeneous biomedical KG
3. Use mixed-curvature geometry for hierarchical relations
4. Enable inductive inference on new glycans

---

## 5. Pseudocode: GlycoPathNet Training Loop

```python
# ============================================================
# GlycoPathNet: Training Loop Pseudocode
# ============================================================

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing

# ----- Stage 1: GlycanEncoder -----
class GlycanTreeEncoder(nn.Module):
    """
    Encodes glycan tree/DAG structure into fixed-dimensional embedding.
    Uses bidirectional tree message passing (bottom-up + top-down).
    """
    def __init__(self, d_mono, d_link, d_hidden, n_layers):
        self.node_embed = nn.Linear(d_mono, d_hidden)
        self.link_embed = nn.Linear(d_link, d_hidden)
        self.msg_net = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.update_gru = nn.GRUCell(d_hidden, d_hidden)
        self.attn_proj = nn.Linear(2 * d_hidden, 1)
        self.readout = nn.Linear(3 * d_hidden, d_hidden)
        self.n_layers = n_layers

    def forward(self, glycan_tree):
        """
        Args:
            glycan_tree: dict with keys:
                - node_features: [m, d_mono]  (monosaccharide features)
                - edge_features: [m-1, d_link] (linkage features)
                - edge_index: [2, m-1]  (parent-child adjacency)
                - root_idx: int
                - topo_order: list[int]  (topological order, leaves first)
        Returns:
            z_g: [d_hidden] glycan embedding
        """
        x = self.node_embed(glycan_tree.node_features)  # [m, d]

        # Bottom-up pass (leaves to root)
        for layer in range(self.n_layers):
            h_new = torch.zeros_like(x)
            for v in glycan_tree.topo_order:  # leaves first
                children = get_children(v, glycan_tree.edge_index)
                if len(children) == 0:
                    h_new[v] = x[v]
                    continue

                # Compute messages from children
                msgs = []
                attn_scores = []
                for u, e_idx in children:
                    link_feat = self.link_embed(glycan_tree.edge_features[e_idx])
                    msg = self.msg_net(torch.cat([x[u], x[v], link_feat]))
                    msgs.append(msg)
                    attn_scores.append(self.attn_proj(torch.cat([x[u], x[v]])))

                # Attention-weighted aggregation
                alphas = torch.softmax(torch.stack(attn_scores), dim=0)
                agg = (alphas * torch.stack(msgs)).sum(dim=0)

                # GRU update
                h_new[v] = self.update_gru(agg, x[v])

            x = h_new

        # Top-down pass (root to leaves) -- analogous, reversed direction
        x = self.top_down_pass(x, glycan_tree)

        # Readout: [mean || max || root]
        z_g = self.readout(torch.cat([
            x.mean(dim=0),
            x.max(dim=0).values,
            x[glycan_tree.root_idx]
        ]))

        return z_g


# ----- Stage 2: PathReasoner (NBFNet-style) -----
class PathReasoner(MessagePassing):
    """
    Generalized Bellman-Ford message passing on the KG.
    Computes path representations from query entity to all entities.
    """
    def __init__(self, d_hidden, n_relations, n_iterations):
        super().__init__(aggr='add')
        self.n_iterations = n_iterations
        # Relation-specific message MLPs
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                nn.Linear(d_hidden, d_hidden)
            ) for _ in range(n_relations)
        ])
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, query_entity, query_relation, entity_embeddings, edge_index, edge_type):
        """
        Args:
            query_entity: int (index of head entity h)
            query_relation: int (index of query relation r)
            entity_embeddings: [n_entities, d] (from GlycanEncoder or learned)
            edge_index: [2, |E|]
            edge_type: [|E|]
        Returns:
            scores: [n_entities] score for each candidate tail entity
        """
        n_entities = entity_embeddings.size(0)

        # Initialize: boundary condition
        h = torch.zeros(n_entities, entity_embeddings.size(1))
        h[query_entity] = entity_embeddings[query_entity]

        # T iterations of Bellman-Ford
        for t in range(self.n_iterations):
            # Message passing
            messages = torch.zeros_like(h)
            for rel_idx in range(len(self.msg_mlps)):
                mask = (edge_type == rel_idx)
                if mask.any():
                    src = edge_index[0, mask]
                    dst = edge_index[1, mask]
                    msg = self.msg_mlps[rel_idx](h[src])
                    messages.index_add_(0, dst, msg)

            # Update
            h = self.update_mlp(torch.cat([h, messages], dim=-1))

        # Score all candidate tails
        r_emb = self.relation_embed(query_relation)  # [d]
        scores = self.score_mlp(torch.cat([h, r_emb.expand_as(h)], dim=-1)).squeeze(-1)

        return scores


# ----- Hyperbolic Component -----
class PoincareDistance(nn.Module):
    """Poincare ball distance for hierarchical relations."""
    def __init__(self, curvature=1.0):
        self.c = curvature

    def forward(self, u, v):
        """Poincare distance between u and v in the Poincare ball."""
        sqrt_c = self.c ** 0.5
        u_norm_sq = (u * u).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
        diff_norm_sq = ((u - v) ** 2).sum(dim=-1, keepdim=True)

        delta = 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq) + 1e-8)
        dist = (1 / sqrt_c) * torch.acosh(1 + self.c * delta)
        return dist.squeeze(-1)

    def exp_map(self, x, v):
        """Exponential map from tangent space at x to Poincare ball."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        lambda_x = 2 / (1 - self.c * x_norm_sq + 1e-8)

        tanh_arg = (self.c ** 0.5) * lambda_x * v_norm / 2
        direction = v / v_norm

        return mobius_add(x, torch.tanh(tanh_arg) * direction / (self.c ** 0.5), self.c)


# ----- Full Model -----
class GlycoPathNet(nn.Module):
    def __init__(self, config):
        self.glycan_encoder = GlycanTreeEncoder(
            d_mono=config.d_mono,
            d_link=config.d_link,
            d_hidden=config.d_hidden,
            n_layers=config.encoder_layers
        )
        self.path_reasoner = PathReasoner(
            d_hidden=config.d_hidden,
            n_relations=config.n_relations,
            n_iterations=config.n_iterations
        )
        self.poincare = PoincareDistance(curvature=config.curvature)

        # Learnable embeddings for non-glycan entities
        self.entity_embed = nn.Embedding(config.n_entities, config.d_hidden)
        self.relation_embed = nn.Embedding(config.n_relations, config.d_hidden)

        # Relation-type-dependent view weights
        self.view_weights = nn.Linear(config.d_hidden, 3)  # alpha, beta, gamma per relation

        # Structural similarity projection
        self.struct_proj = nn.ModuleDict({
            str(r): nn.Linear(config.d_hidden, config.d_hidden)
            for r in config.structural_relations  # child_of, has_motif
        })

    def get_entity_embeddings(self, glycan_trees, entity_types):
        """Compute entity embeddings: GlycanEncoder for glycans, lookup for others."""
        embeddings = self.entity_embed.weight.clone()

        for idx, glycan_tree in glycan_trees.items():
            embeddings[idx] = self.glycan_encoder(glycan_tree)

        return embeddings

    def score(self, h_idx, r_idx, t_idx, entity_embeddings, edge_index, edge_type):
        """Multi-view scoring function."""
        r_emb = self.relation_embed(r_idx)
        weights = torch.softmax(self.view_weights(r_emb), dim=-1)
        alpha, beta, gamma = weights[0], weights[1], weights[2]

        # View 1: Path-based score
        s_path = self.path_reasoner(h_idx, r_idx, entity_embeddings, edge_index, edge_type)
        s_path = s_path[t_idx]

        # View 2: Structural similarity (glycan-glycan relations only)
        e_h = entity_embeddings[h_idx]
        e_t = entity_embeddings[t_idx]
        if r_idx in self.struct_proj:
            s_struct = torch.cosine_similarity(
                e_h, self.struct_proj[str(r_idx)](e_t), dim=-1
            )
        else:
            s_struct = torch.zeros_like(s_path)

        # View 3: Hyperbolic distance (hierarchical relations)
        s_hyp = -self.poincare(e_h, e_t)

        return alpha * s_path + beta * s_struct + gamma * s_hyp

    def compute_loss(self, pos_triples, neg_triples, entity_embeddings, edge_index, edge_type):
        """Self-adversarial negative sampling loss."""
        # Positive scores
        h, r, t = pos_triples.T
        pos_scores = self.score(h, r, t, entity_embeddings, edge_index, edge_type)

        # Negative scores with self-adversarial weighting
        h_neg, r_neg, t_neg = neg_triples.T
        neg_scores = self.score(h_neg, r_neg, t_neg, entity_embeddings, edge_index, edge_type)

        # Self-adversarial weights
        with torch.no_grad():
            neg_weights = torch.softmax(neg_scores * self.adv_temperature, dim=0)

        # BCE loss
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
        neg_loss = -(neg_weights * torch.log(torch.sigmoid(-neg_scores) + 1e-8)).sum()

        return pos_loss + neg_loss


# ============================================================
# Training Loop
# ============================================================
def train_glycopathnet(model, kg_data, glycan_trees, config):
    """
    Main training loop for GlycoPathNet.

    Args:
        model: GlycoPathNet instance
        kg_data: dict with edge_index, edge_type, train/val/test splits
        glycan_trees: dict mapping glycan entity indices to tree structures
        config: training configuration
    """
    optimizer = torch.optim.Adam([
        {'params': model.glycan_encoder.parameters(), 'lr': config.lr_encoder},
        {'params': model.path_reasoner.parameters(), 'lr': config.lr_reasoner},
        {'params': model.entity_embed.parameters(), 'lr': config.lr_embed},
        {'params': model.relation_embed.parameters(), 'lr': config.lr_embed},
        {'params': model.poincare.parameters(), 'lr': config.lr_hyp},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.warmup_epochs, T_mult=2
    )

    best_mrr = 0
    patience_counter = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0

        # Step 1: Compute entity embeddings (glycan encoder + lookup)
        entity_embeddings = model.get_entity_embeddings(glycan_trees, kg_data.entity_types)

        # Step 2: Sample training triples in mini-batches
        for batch in sample_training_batches(kg_data.train_triples, config.batch_size):
            optimizer.zero_grad()

            # Step 3: Generate negative samples (type-constrained)
            neg_triples = type_constrained_negative_sampling(
                batch,
                kg_data.entity_types,
                kg_data.relation_domains,  # valid source types per relation
                kg_data.relation_ranges,   # valid target types per relation
                n_neg=config.n_negative_samples
            )

            # Step 4: Compute loss
            loss = model.compute_loss(
                pos_triples=batch,
                neg_triples=neg_triples,
                entity_embeddings=entity_embeddings,
                edge_index=kg_data.edge_index,
                edge_type=kg_data.edge_type
            )

            # Step 5: Add regularization
            # Contrastive loss on glycan structural embeddings
            struct_loss = glycan_contrastive_loss(
                entity_embeddings, glycan_trees, kg_data.motif_sharing_pairs
            )
            # Hyperbolic regularization
            hyp_reg = hyperbolic_regularization(entity_embeddings, model.poincare.c)

            total_loss = loss + config.lambda_struct * struct_loss + config.lambda_hyp * hyp_reg

            # Step 6: Backward pass
            total_loss.backward()

            # Gradient clipping (important for hyperbolic embeddings)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            # Project hyperbolic embeddings back to Poincare ball
            with torch.no_grad():
                project_to_poincare_ball(model.entity_embed.weight, model.poincare.c)

            epoch_loss += total_loss.item()

        scheduler.step()

        # Step 7: Validation
        if epoch % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                entity_embeddings = model.get_entity_embeddings(glycan_trees, kg_data.entity_types)
                metrics = evaluate(
                    model, kg_data.val_triples, entity_embeddings,
                    kg_data.edge_index, kg_data.edge_type,
                    metrics=['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
                )

            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                  f"MRR={metrics['MRR']:.4f}, H@10={metrics['Hits@10']:.4f}")

            # Early stopping
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, config.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    return model


# ============================================================
# Helper Functions
# ============================================================
def type_constrained_negative_sampling(pos_triples, entity_types,
                                        relation_domains, relation_ranges, n_neg):
    """
    Generate negative samples respecting entity type constraints.
    For relation r with domain D and range R:
      - Corrupt head: sample h' from entities of type in D
      - Corrupt tail: sample t' from entities of type in R
    """
    neg_triples = []
    for h, r, t in pos_triples:
        for _ in range(n_neg):
            if torch.rand(1) < 0.5:
                # Corrupt head
                valid_heads = get_entities_of_types(entity_types, relation_domains[r])
                h_neg = valid_heads[torch.randint(len(valid_heads), (1,))]
                neg_triples.append((h_neg, r, t))
            else:
                # Corrupt tail
                valid_tails = get_entities_of_types(entity_types, relation_ranges[r])
                t_neg = valid_tails[torch.randint(len(valid_tails), (1,))]
                neg_triples.append((h, r, t_neg))

    return torch.tensor(neg_triples)


def glycan_contrastive_loss(entity_embeddings, glycan_trees, motif_pairs, temperature=0.1):
    """
    Contrastive loss encouraging glycans sharing motifs to have similar embeddings.
    Uses InfoNCE-style loss.
    """
    loss = 0
    for (g1, g2) in motif_pairs:  # g1, g2 share at least one motif
        z1 = entity_embeddings[g1]
        z2 = entity_embeddings[g2]
        sim = torch.cosine_similarity(z1, z2, dim=0) / temperature
        # Negative pairs: random glycans
        neg_sims = torch.cosine_similarity(
            z1.unsqueeze(0), entity_embeddings[random_glycan_indices], dim=1
        ) / temperature
        loss += -sim + torch.logsumexp(torch.cat([sim.unsqueeze(0), neg_sims]), dim=0)

    return loss / max(len(motif_pairs), 1)
```

---

## 6. Implementation Recommendations

### 6.1 Framework Dependencies

| Component | Recommended Library | Reason |
|-----------|-------------------|--------|
| Graph operations | PyTorch Geometric (PyG) | De facto standard; supports heterogeneous graphs |
| KG embedding baselines | PyKEEN | Comprehensive KGE library with 40+ models |
| Glycan parsing | glycowork | WURCS/IUPAC to graph conversion built-in |
| Hyperbolic operations | geoopt | Riemannian optimization in PyTorch |
| Path-based reasoning | TorchDrug / NBFNet | Official NBFNet implementation |
| Experiment tracking | Weights & Biases | Hyperparameter sweep support |

### 6.2 Recommended Hyperparameters (Initial)

```yaml
model:
  d_hidden: 256
  d_mono: 64        # monosaccharide feature dim
  d_link: 16        # linkage feature dim
  encoder_layers: 3  # GlycanEncoder depth
  n_iterations: 6    # Bellman-Ford iterations (path length)
  curvature: 1.0     # Poincare ball curvature (learnable)

training:
  lr_encoder: 1e-3
  lr_reasoner: 5e-4
  lr_embed: 1e-3
  lr_hyp: 1e-2       # higher LR for Riemannian optimizer
  batch_size: 1024
  n_negative_samples: 256
  adv_temperature: 1.0
  max_grad_norm: 1.0
  max_epochs: 500
  patience: 30
  eval_every: 5

regularization:
  lambda_struct: 0.1
  lambda_hyp: 0.01
  weight_decay: 1e-5

negative_sampling:
  strategy: "type_constrained_self_adversarial"
  n_neg: 256
```

### 6.3 Phased Implementation Plan

**Phase 1 (Baseline)**: Implement RotatE and CompGCN baselines on glycoMusubi using PyKEEN
- Purpose: Establish baseline performance numbers
- Effort: 1-2 weeks

**Phase 2 (GlycanEncoder)**: Implement GlycanTreeEncoder
- Parse WURCS strings to tree graphs using glycowork
- Train GlycanEncoder standalone on glycan classification tasks (GlycanML benchmark)
- Validate that structure-derived embeddings capture known glycan similarities
- Effort: 2-3 weeks

**Phase 3 (PathReasoner)**: Implement NBFNet-based PathReasoner
- Adapt BioPathNet/NBFNet to glycoMusubi schema
- Replace glycan entity embeddings with GlycanEncoder output
- Evaluate link prediction performance
- Effort: 2-3 weeks

**Phase 4 (Full GlycoPathNet)**: Integrate hyperbolic scoring + multi-view fusion
- Add Poincare distance scoring for hierarchical relations
- Implement multi-view fusion with learned weights
- Add contrastive structural loss
- Effort: 2-3 weeks

**Phase 5 (Ablation & Analysis)**: Comprehensive evaluation
- Ablation study: path-only vs. struct-only vs. hyp-only vs. full
- Interpretability analysis: extract and visualize prediction paths
- Comparison with all baselines
- Effort: 2-3 weeks

---

## 7. References

1. Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" ICLR 2019
2. Chao et al. "PairRE: Knowledge Graph Embeddings via Paired Relation Vectors" ACL 2021
3. Li et al. "HousE: Knowledge Graph Embedding with Householder Parameterization" ICML 2022
4. Vashishth et al. "Composition-based Multi-Relational Graph Convolutional Networks" ICLR 2020
5. Zhu et al. "Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction" NeurIPS 2021
6. Galkin et al. "Towards Foundation Models for Knowledge Graph Reasoning" (ULTRA) ICLR 2024
7. Nusser et al. "BioPathNet: Enhancing Link Prediction in Biomedical Knowledge Graphs" Nature Biomedical Engineering 2025
8. Burkholz et al. "Using graph convolutional neural networks to learn a representation for glycans" (SweetNet) Cell Reports 2021
9. Joeres & Bojar. "Higher-Order Message Passing for Glycan Representation Learning" (GIFFLAR) arXiv 2024
10. "Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training" (GlycanAA) ICML 2025
11. Chen et al. "Enhancing Hyperbolic KGE via Lorentz Transformations" (LorentzKG) ACL Findings 2024
12. "Fully Hyperbolic Rotation for Knowledge Graph Embedding" arXiv 2024
13. Chami et al. "Hyperbolic Hierarchical Knowledge Graph Embeddings" (HEM) 2024
14. Bojar et al. "Glycowork: A Python package for glycan data science and machine learning" Glycobiology 2021
15. Galkin et al. "A Foundation Model for Zero-shot Logical Query Reasoning" (UltraQuery) NeurIPS 2024

---

## Appendix A: Monosaccharide Feature Encoding

For the GlycanEncoder, each monosaccharide node v_i is encoded with the following features:

| Feature | Encoding | Dimension |
|---------|----------|-----------|
| Monosaccharide type (Glc, Gal, Man, GlcNAc, ...) | One-hot or learned embedding (64 common types) | 32 |
| Ring form (pyranose, furanose) | One-hot | 2 |
| Anomeric configuration (alpha, beta) | One-hot | 2 |
| Modifications (sulfation, phosphorylation, ...) | Multi-hot | 16 |
| Absolute configuration (D, L) | Binary | 1 |
| **Total** | | **53** -> projected to d_mono=64 |

Linkage edges encode:
| Feature | Encoding | Dimension |
|---------|----------|-----------|
| Carbon position of parent (1-6) | One-hot | 6 |
| Carbon position of child (1-6) | One-hot | 6 |
| Linkage confidence | Scalar | 1 |
| **Total** | | **13** -> projected to d_link=16 |

## Appendix B: Relation Type Categorization for Multi-View Scoring

| Relation | Primary View | Secondary View | Rationale |
|----------|-------------|----------------|-----------|
| inhibits | S_path | - | Multi-hop pharmacological reasoning |
| has_glycan | S_path | S_struct | Glycan structure determines binding |
| associated_with_disease | S_path | - | Complex multi-hop associations |
| has_variant | S_path | - | Sequence-level; path captures gene-disease chains |
| produced_by / consumed_by | S_path | S_struct | Enzyme specificity depends on glycan structure |
| child_of (subsumption) | S_hyp | S_struct | **Hierarchical**: parent-child tree structure |
| has_motif | S_hyp | S_struct | **Hierarchical**: part-whole containment |
| catalyzed_by | S_path | - | Enzymatic pathway reasoning |
| has_site | S_path | - | Protein-site relationship |
| ptm_crosstalk | S_path | - | Cross-talk pathway discovery |
| has_product | S_path | - | Reaction product chain |
| associated_with_disease (enzyme) | S_path | - | Enzyme-disease multi-hop |
