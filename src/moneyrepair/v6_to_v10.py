"""Advanced modeling architectures for MoneyRepair (v6 to v10).

This module implements the architectural schemas and core code for the v6-v10
evolution of the banknote reconstruction system:
- v6: Graph Neural Assembly (learned edge probability + soft assignment + relaxed exact cover)
- v7: Energy-Based Assembly (global energy scorer + MCMC simulated annealing)
- v8: Graph Diffusion Assembly (forward edge corruption + reverse denoising loops)
- v9: Neural ILP Solver (differentiable set cover optimization via Transformer + simplex projection)
- v10: Latent Structure World Model (inverse graphics structure inference)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic
from typing import Any, Iterable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Define simple placeholders so code compiles/imports without PyTorch
    class nn_Module:
        def __init__(self, *args, **kwargs): pass
    torch = None
    nn = nn_Module()
    nn.Module = nn_Module
    F = None

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# =====================================================================
# v6: Graph Neural Assembly System (GNAS)
# =====================================================================

V6_TO_V10_ARCHITECTURES = (
    "v6_gnn_soft_cover",
    "v7_energy_mcmc",
    "v8_diffusion",
    "v9_neural_ilp",
    "v10_latent_world",
)

class FragmentEncoder(nn.Module):
    """Encodes fragment images and masks into high-dimensional embeddings."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        if HAS_TORCH and HAS_TORCHVISION:
            # Load a pretrained ResNet backbone and adjust input/output channels
            resnet = models.resnet18(pretrained=False)
            # Input: 4 channels (RGB image + 1-channel binary mask)
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(512, embedding_dim)
        elif HAS_TORCH:
            # Fallback custom ConvNet if torchvision is missing
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(128, embedding_dim)

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, 3, H, W)
            masks: Tensor of shape (B, 1, H, W)
        Returns:
            Embeddings of shape (B, embedding_dim)
        """
        x = torch.cat([images, masks], dim=1) # (B, 4, H, W)
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)


class GraphAttentionLayer(nn.Module):
    """Pure PyTorch GAT Layer to run on standard matrices without PyG dependencies."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature matrix of shape (N, in_features)
            adj: Adjacency matrix of shape (N, N)
        """
        Wh = torch.mm(h, self.W) # (N, out_features)
        N = Wh.size(0)

        # Concatenate features for all node pairs
        a_input = torch.cat([
            Wh.repeat(1, N).view(N * N, -1),
            Wh.repeat(N, 1)
        ], dim=1).view(N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, N)

        # Mask attention coefficients using adjacency structure
        # Add self-loops to prevent uniform/NaN scores on zero-degree rows
        adj_with_self = adj.clone()
        adj_with_self.fill_diagonal_(1.0)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_with_self > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        return h_prime


class EdgeModel(nn.Module):
    """Siamese classifier predicting probability of connection between fragments."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # Input features are symmetric (sum, absolute difference, product) -> 3 * D
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, fi: torch.Tensor, fj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fi: Embeddings of first fragment (B, D)
            fj: Embeddings of second fragment (B, D)
        """
        sum_feat = fi + fj
        diff_feat = torch.abs(fi - fj)
        prod_feat = fi * fj
        x = torch.cat([sum_feat, diff_feat, prod_feat], dim=-1)
        return torch.sigmoid(self.mlp(x))


class AssemblyGNN(nn.Module):
    """Refines fragment representations globally using GAT convolution."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.conv1 = GraphAttentionLayer(embedding_dim, embedding_dim, concat=True)
        self.conv2 = GraphAttentionLayer(embedding_dim, embedding_dim, concat=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, D)
            adj: Adjacency matrix (N, N)
        """
        h = self.conv1(x, adj)
        return self.conv2(h, adj)


def sinkhorn_soft_assignment(P: torch.Tensor, tau: float = 0.1, iterations: int = 50, eps: float = 1e-9) -> torch.Tensor:
    """Continuous assignment matrix relaxation using standard Sinkhorn iterations."""
    z = P / tau
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
        z = z - torch.logsumexp(z, dim=0, keepdim=True)
    return torch.exp(z)


def compute_v6_loss(
    P: torch.Tensor,
    A: torch.Tensor,
    y_edge: torch.Tensor,
    *,
    lambda_edge: float = 1.0,
    lambda_contrast: float = 2.0,
    lambda_cover: float = 1.5,
    lambda_entropy: float = 0.1,
    margin: float = 0.5,
) -> torch.Tensor:
    """Computes the combined loss for Sinkhorn soft matching / pair assignment relaxation."""
    # 1. Edge Loss (BCE)
    loss_edge = F.binary_cross_entropy(P, y_edge)

    # 2. Contrastive Graph Loss
    pos_mask = (y_edge > 0.5)
    neg_mask = (y_edge <= 0.5)
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        P_pos = P[pos_mask].mean()
        P_neg = P[neg_mask].mean()
        loss_contrast = F.relu(margin - P_pos + P_neg)
    else:
        loss_contrast = torch.tensor(0.0, device=P.device)

    # 3. Coverage Loss: each node is soft-assigned exactly once
    loss_cover = torch.mean((A.sum(dim=1) - 1.0) ** 2)

    # 4. Entropy Loss: prevents trivial flat matching solutions
    loss_entropy = -torch.mean(A * torch.log(A + 1e-9))

    return (
        lambda_edge * loss_edge
        + lambda_contrast * loss_contrast
        + lambda_cover * loss_cover
        + lambda_entropy * loss_entropy
    )


def candidate_soft_exact_cover_loss(
    candidate_fragment_incidence: torch.Tensor,
    selection_probs: torch.Tensor,
    *,
    lambda_cover: float = 1.0,
    lambda_overlap: float = 1.0,
    lambda_missing: float = 1.0,
    lambda_entropy: float = 0.01,
) -> torch.Tensor:
    """Differentiable candidate-level exact-cover relaxation.

    Args:
        candidate_fragment_incidence: Binary/soft matrix with shape
            ``(num_candidates, num_fragments)``. ``1`` means a candidate uses a
            fragment.
        selection_probs: Soft candidate selection variables with shape
            ``(num_candidates,)`` or ``(num_candidates, 1)``.

    The old Sinkhorn matrix is a pair-assignment relaxation. This loss models
    the actual exact-cover surface: selected candidate assemblies should cover
    each fragment exactly once.
    """

    if selection_probs.ndim == 2:
        selection_probs = selection_probs.squeeze(-1)
    if candidate_fragment_incidence.ndim != 2:
        raise ValueError("candidate_fragment_incidence must be a 2-D matrix")
    if selection_probs.ndim != 1:
        raise ValueError("selection_probs must be a vector or a single-column matrix")
    if candidate_fragment_incidence.shape[0] != selection_probs.shape[0]:
        raise ValueError("candidate count must match selection probability count")

    coverage = candidate_fragment_incidence.T @ selection_probs
    loss_cover = torch.mean((coverage - 1.0) ** 2)
    loss_overlap = torch.mean(F.relu(coverage - 1.0) ** 2)
    loss_missing = torch.mean(F.relu(1.0 - coverage) ** 2)
    loss_entropy = -torch.mean(
        selection_probs * torch.log(selection_probs + 1e-9)
        + (1.0 - selection_probs) * torch.log(1.0 - selection_probs + 1e-9)
    )
    return (
        lambda_cover * loss_cover
        + lambda_overlap * loss_overlap
        + lambda_missing * loss_missing
        + lambda_entropy * loss_entropy
    )


@dataclass(frozen=True)
class AssemblyTrainingSample:
    """Synthetic latent assembly supervision bundle for v6 training smoke tests."""

    node_embeddings: Any
    clean_adj: Any
    edge_labels: Any
    candidate_fragment_incidence: Any
    serial_labels: Any
    hard_negative_edges: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class V6TrainingSmokeConfig:
    nodes: int = 8
    pieces_per_note: int = 4
    embedding_dim: int = 32
    seed: int = 7
    feature_noise: float = 0.10
    serial_dropout: float = 0.30
    hard_negative_top_k: int = 6
    steps: int = 20
    lr: float = 0.01
    lambda_contrast: float = 2.0
    lambda_serial: float = 0.2


def mine_hard_negative_edges(
    edge_probs: torch.Tensor,
    clean_adj: torch.Tensor,
    *,
    top_k: int = 6,
) -> tuple[tuple[int, int, float], ...]:
    """Return the highest-scoring false edges for adversarial refinement."""

    _require_torch()
    if top_k <= 0:
        return ()
    scores = edge_probs.detach().float()
    labels = clean_adj.detach().float()
    rows, cols = torch.triu_indices(scores.shape[0], scores.shape[1], offset=1)
    false_mask = labels[rows, cols] <= 0.5
    rows = rows[false_mask]
    cols = cols[false_mask]
    if rows.numel() == 0:
        return ()
    values = scores[rows, cols]
    count = min(top_k, values.numel())
    top_values, top_indices = torch.topk(values, k=count)
    return tuple(
        (int(rows[index]), int(cols[index]), float(top_values[pos].item()))
        for pos, index in enumerate(top_indices)
    )


def edge_probability_matrix(edge_model: EdgeModel, embeddings: torch.Tensor) -> torch.Tensor:
    """Evaluate an EdgeModel on all ordered node pairs as a symmetric matrix."""

    _require_torch()
    count, dim = embeddings.shape
    left = embeddings.unsqueeze(1).repeat(1, count, 1).reshape(-1, dim)
    right = embeddings.unsqueeze(0).repeat(count, 1, 1).reshape(-1, dim)
    probs = edge_model(left, right).reshape(count, count)
    probs = 0.5 * (probs + probs.T)
    eye = torch.eye(count, device=embeddings.device, dtype=probs.dtype)
    return probs * (1.0 - eye)


def collapse_diagnostics(edge_probs: torch.Tensor, serial_labels: torch.Tensor | None = None) -> dict[str, float]:
    """Measure common graph-learning collapse modes without claiming accuracy."""

    _require_torch()
    probs = edge_probs.detach().float().clamp(1e-6, 1.0 - 1e-6)
    count = probs.shape[0]
    offdiag = probs[~torch.eye(count, dtype=torch.bool, device=probs.device)]
    entropy = -(offdiag * torch.log(offdiag) + (1.0 - offdiag) * torch.log(1.0 - offdiag)).mean()
    row_mass = probs.sum(dim=1)
    all_to_one_score = row_mass.max() / row_mass.sum().clamp_min(1e-6)
    diagnostics = {
        "edge_density": float(offdiag.mean().item()),
        "edge_entropy": float(entropy.item()),
        "all_to_one_score": float(all_to_one_score.item()),
    }
    if serial_labels is not None:
        known = serial_labels >= 0
        same = (serial_labels[:, None] == serial_labels[None, :]) & known[:, None] & known[None, :]
        different = (serial_labels[:, None] != serial_labels[None, :]) & known[:, None] & known[None, :]
        diag = torch.eye(count, dtype=torch.bool, device=probs.device)
        same = same & ~diag
        different = different & ~diag
        same_mean = probs[same].mean() if same.any() else torch.tensor(0.0, device=probs.device)
        diff_mean = probs[different].mean() if different.any() else torch.tensor(0.0, device=probs.device)
        diagnostics["serial_same_mean"] = float(same_mean.item())
        diagnostics["serial_different_mean"] = float(diff_mean.item())
        diagnostics["serial_gap"] = float((same_mean - diff_mean).item())
    return diagnostics


def make_assembly_training_sample(
    *,
    nodes: int = 8,
    pieces_per_note: int = 4,
    embedding_dim: int = 32,
    seed: int = 7,
    feature_noise: float = 0.10,
    serial_dropout: float = 0.30,
    hard_negative_top_k: int = 6,
) -> AssemblyTrainingSample:
    """Generate a controllable latent assembly sample with labels and hard negatives."""

    _require_torch()
    if not (0.0 <= serial_dropout <= 1.0):
        raise ValueError("serial_dropout must be in [0, 1]")
    embeddings, clean_adj = _synthetic_clean_graph(nodes, pieces_per_note, embedding_dim, seed)
    torch.manual_seed(seed + 17)
    embeddings = embeddings + torch.randn_like(embeddings) * feature_noise
    serial_labels = torch.full((nodes,), -1, dtype=torch.long)
    for group_index, start in enumerate(range(0, nodes, pieces_per_note)):
        end = min(nodes, start + pieces_per_note)
        serial_labels[start:end] = group_index
    if serial_dropout > 0.0:
        keep = torch.rand(nodes) >= serial_dropout
        serial_labels = torch.where(keep, serial_labels, torch.full_like(serial_labels, -1))
    candidate_incidence = _candidate_incidence(nodes, pieces_per_note)
    normed = F.normalize(embeddings, dim=1)
    similarity_probs = ((normed @ normed.T) + 1.0) * 0.5
    similarity_probs.fill_diagonal_(0.0)
    hard_negatives = mine_hard_negative_edges(similarity_probs, clean_adj, top_k=hard_negative_top_k)
    return AssemblyTrainingSample(
        node_embeddings=embeddings,
        clean_adj=clean_adj,
        edge_labels=clean_adj.clone(),
        candidate_fragment_incidence=candidate_incidence,
        serial_labels=serial_labels,
        hard_negative_edges=hard_negatives,
    )


def _serial_constraint_loss(edge_probs: torch.Tensor, serial_labels: torch.Tensor) -> torch.Tensor:
    known = serial_labels >= 0
    different = (serial_labels[:, None] != serial_labels[None, :]) & known[:, None] & known[None, :]
    different = different & ~torch.eye(edge_probs.shape[0], dtype=torch.bool, device=edge_probs.device)
    if not different.any():
        return torch.tensor(0.0, device=edge_probs.device)
    return edge_probs[different].mean()


def _hard_negative_margin_loss(
    edge_probs: torch.Tensor,
    clean_adj: torch.Tensor,
    hard_negative_edges: tuple[tuple[int, int, float], ...],
    *,
    margin: float = 0.25,
) -> torch.Tensor:
    pos = edge_probs[clean_adj > 0.5]
    if pos.numel() == 0 or not hard_negative_edges:
        return torch.tensor(0.0, device=edge_probs.device)
    hard_values = torch.stack([edge_probs[left, right] for left, right, _score in hard_negative_edges])
    return F.relu(margin - pos.mean() + hard_values.mean())


def run_v6_training_smoke(config: V6TrainingSmokeConfig | None = None) -> dict[str, Any]:
    """Run a tiny supervised v6 training loop with hard-negative and collapse metrics."""

    _require_torch()
    config = config or V6TrainingSmokeConfig()
    torch.manual_seed(config.seed)
    sample = make_assembly_training_sample(
        nodes=config.nodes,
        pieces_per_note=config.pieces_per_note,
        embedding_dim=config.embedding_dim,
        seed=config.seed,
        feature_noise=config.feature_noise,
        serial_dropout=config.serial_dropout,
        hard_negative_top_k=config.hard_negative_top_k,
    )
    model = EdgeModel(embedding_dim=config.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    def loss_and_metrics() -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        probs = edge_probability_matrix(model, sample.node_embeddings)
        assignment = sinkhorn_soft_assignment(probs, tau=0.2, iterations=30)
        supervised = compute_v6_loss(
            probs.reshape(-1, 1),
            assignment,
            sample.edge_labels.reshape(-1, 1),
            lambda_contrast=config.lambda_contrast,
            lambda_cover=0.1,
        )
        serial_loss = _serial_constraint_loss(probs, sample.serial_labels)
        hard_loss = _hard_negative_margin_loss(probs, sample.clean_adj, sample.hard_negative_edges)
        loss = supervised + config.lambda_serial * serial_loss + hard_loss
        metrics = _edge_proxy_metrics(probs, sample.clean_adj)
        metrics.update(collapse_diagnostics(probs, sample.serial_labels))
        metrics["serial_constraint_loss"] = float(serial_loss.item())
        metrics["hard_negative_margin_loss"] = float(hard_loss.item())
        metrics["loss"] = float(loss.detach().item())
        return loss, probs, metrics

    initial_loss, initial_probs, initial_metrics = loss_and_metrics()
    loss_curve = [float(initial_loss.detach().item())]
    for _step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _probs, _metrics = loss_and_metrics()
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.detach().item()))

    final_loss, final_probs, final_metrics = loss_and_metrics()
    return {
        "config": {
            **config.__dict__,
            "trained_steps": config.steps,
            "stage": "synthetic_latent_edge_pretrain",
        },
        "initial": {
            "loss": float(initial_loss.detach().item()),
            "metrics": initial_metrics,
            "hard_negative_edges": sample.hard_negative_edges,
        },
        "final": {
            "loss": float(final_loss.detach().item()),
            "metrics": final_metrics,
            "hard_negative_edges": mine_hard_negative_edges(
                final_probs,
                sample.clean_adj,
                top_k=config.hard_negative_top_k,
            ),
        },
        "loss_curve": loss_curve,
        "improved": float(final_loss.detach().item()) < float(initial_loss.detach().item()),
        "collapse_diagnostics": {
            "initial": collapse_diagnostics(initial_probs, sample.serial_labels),
            "final": collapse_diagnostics(final_probs, sample.serial_labels),
        },
    }


# =====================================================================
# v7: Energy-Based Assembly Model (EBM)
# =====================================================================

class EnergyNetwork(nn.Module):
    """Energy-Based Model (EBM) scoring structural harmony of proposed assemblies."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, node_embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Computes the scalar global energy E(G) for assembly graph adj.
        
        Lower energy indicates a more stable / compatible reconstruction structure.
        """
        N = node_embeddings.size(0)
        emb_i = node_embeddings.unsqueeze(1).repeat(1, N, 1) # (N, N, D)
        emb_j = node_embeddings.unsqueeze(0).repeat(N, 1, 1) # (N, N, D)
        pair_features = torch.cat([emb_i, emb_j], dim=-1) # (N, N, 2D)
        
        edge_energies = self.mlp(pair_features).squeeze(-1) # (N, N)
        # Sum energies over active proposed edges
        total_energy = torch.sum(adj * edge_energies) / 2.0
        return total_energy


def mcmc_annealing_search(
    energy_net: EnergyNetwork,
    node_embeddings: torch.Tensor,
    initial_adj: torch.Tensor,
    *,
    steps: int = 100,
    temp_start: float = 10.0,
    temp_end: float = 0.1,
) -> torch.Tensor:
    """MCMC simulated annealing sampler over the space of adjacency structures."""
    current_adj = initial_adj.clone()
    current_energy = energy_net(node_embeddings, current_adj).item()
    N = initial_adj.size(0)
    
    best_adj = current_adj.clone()
    best_energy = current_energy
    
    for step in range(steps):
        t = temp_start * (temp_end / temp_start) ** (step / steps)
        
        # Propose a local structure mutation: flip a random edge (i, j)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        if i == j:
            continue
            
        proposed_adj = current_adj.clone()
        proposed_adj[i, j] = 1.0 - proposed_adj[i, j]
        proposed_adj[j, i] = proposed_adj[i, j]
        
        proposed_energy = energy_net(node_embeddings, proposed_adj).item()
        
        # Metropolis acceptance rule
        dE = proposed_energy - current_energy
        if dE < 0 or np.exp(-dE / t) > np.random.rand():
            current_adj = proposed_adj
            current_energy = proposed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = current_adj.clone()
                
    return best_adj


# =====================================================================
# v8: Graph Diffusion Assembly Model
# =====================================================================

class DenoisingEdgeModel(nn.Module):
    """Predicts clean edge probability from a noisy graph at time step t."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # Input features: emb_i, emb_j, noisy_adj value, time step -> 2 * D + 2
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, node_embeddings: torch.Tensor, noisy_adj: torch.Tensor, t: float) -> torch.Tensor:
        N = node_embeddings.size(0)
        emb_i = node_embeddings.unsqueeze(1).repeat(1, N, 1) # (N, N, D)
        emb_j = node_embeddings.unsqueeze(0).repeat(N, 1, 1) # (N, N, D)
        
        # Condition on current noisy adjacency edge state (N, N, 1)
        edge_state = noisy_adj.unsqueeze(-1)
        
        # Broadcast time step
        time_tensor = torch.tensor(t, device=node_embeddings.device).view(1, 1, 1).repeat(N, N, 1)
        x = torch.cat([emb_i, emb_j, edge_state, time_tensor], dim=-1) # (N, N, 2D + 2)
        
        pred_clean = torch.sigmoid(self.mlp(x).squeeze(-1))
        return pred_clean


class GraphDiffusionAssembly:
    """Manages forward noise corruption and reverse denoising generation of assemblies."""

    def __init__(self, model: DenoisingEdgeModel, beta_start: float = 0.001, beta_end: float = 0.02, num_steps: int = 50):
        self.model = model
        self.num_steps = num_steps
        
        if HAS_TORCH:
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, clean_adj: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupts true assembly G_0 to a noisy state G_t."""
        alpha_bar = self.alphas_cumprod[t]
        flip_prob = 0.5 * (1.0 - math.sqrt(alpha_bar))
        
        # Mirroring and preventing diagonal flipping for undirected graphs
        N = clean_adj.size(0)
        triu_indices = torch.triu_indices(N, N, offset=1)
        
        noisy_adj = clean_adj.clone()
        num_edges = triu_indices.size(1)
        noise_mask_triu = torch.rand(num_edges, device=clean_adj.device) < flip_prob
        
        rows, cols = triu_indices
        noisy_adj[rows, cols] = torch.where(noise_mask_triu, 1.0 - clean_adj[rows, cols], clean_adj[rows, cols])
        noisy_adj[cols, rows] = noisy_adj[rows, cols]
        noisy_adj.fill_diagonal_(0.0)
        
        noise_mask = torch.zeros_like(clean_adj, dtype=torch.bool)
        noise_mask[rows, cols] = noise_mask_triu
        noise_mask[cols, rows] = noise_mask_triu
        return noisy_adj, noise_mask

    def reverse_diffusion(self, node_embeddings: torch.Tensor, noisy_adj: torch.Tensor) -> torch.Tensor:
        """Denoises a random initial graph back to its clean global state."""
        current_adj = noisy_adj.clone()
        for t in reversed(range(self.num_steps)):
            time_val = float(t) / self.num_steps
            pred_clean = self.model(node_embeddings, current_adj, time_val)
            
            # Gradually update structure towards predictions
            alpha = self.alphas[t]
            current_adj = alpha * current_adj + (1.0 - alpha) * pred_clean
            
            # Symmetrize and zero-diagonal to preserve undirected invariants
            current_adj = 0.5 * (current_adj + current_adj.T)
            current_adj.fill_diagonal_(0.0)
        return current_adj


# =====================================================================
# v9: Neural ILP Solver (Combinatorial Optimizer)
# =====================================================================

class NeuralILPSolver(nn.Module):
    """Transformer-based solver to predict optimal set-cover selections."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        if HAS_TORCH:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.proj = nn.Linear(embedding_dim, 1)

    def forward(self, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """Predicts soft selection flags for candidate assemblies.
        
        Args:
            candidate_embeddings: Shape (M, embedding_dim)
        Returns:
            Continuous selection variables in [0, 1] of shape (M, 1)
        """
        # Batch dimension setup
        x = candidate_embeddings.unsqueeze(0) # (1, M, D)
        h = self.transformer(x).squeeze(0) # (M, D)
        
        scores = self.proj(h) # (M, 1)
        return torch.sigmoid(scores)


# =====================================================================
# v10: Latent Structure World Model
# =====================================================================

class LatentWorldEncoder(nn.Module):
    """Encodes multiple banknote fragments into a unified scene latent vector z."""

    def __init__(self, embedding_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, latent_dim, batch_first=True)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, fragment_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fragment_embeddings: (N, D)
        Returns:
            mu, logvar for sampling latent state z
        """
        _, h = self.gru(fragment_embeddings.unsqueeze(0))
        h = h.squeeze(0) # (1, latent_dim)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class LatentWorldDecoder(nn.Module):
    """Generates coordinate positions and global connection graphs from scene latent z."""

    def __init__(self, latent_dim: int = 128, num_nodes: int = 20):
        super().__init__()
        self.num_nodes = num_nodes
        self.fc_adj = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * num_nodes)
        )
        self.fc_coords = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * 2) # X, Y coordinates per fragment node
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Latent vector (1, latent_dim)
        Returns:
            adjacency matrix (num_nodes, num_nodes) and coordinates (num_nodes, 2)
        """
        adj_flat = self.fc_adj(z)
        adj = torch.sigmoid(adj_flat.view(self.num_nodes, self.num_nodes))
        
        coords_flat = self.fc_coords(z)
        coords = coords_flat.view(self.num_nodes, 2)
        return adj, coords


# =====================================================================
# v6-v10 Architecture Smoke Comparison
# =====================================================================

def _require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError("v6-v10 architecture comparison requires optional dependency torch")


def _synthetic_clean_graph(nodes: int, pieces_per_note: int, embedding_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic latent assembly graph for architecture smoke tests."""

    _require_torch()
    if nodes < 2:
        raise ValueError("nodes must be at least 2")
    if pieces_per_note < 2:
        raise ValueError("pieces_per_note must be at least 2")
    torch.manual_seed(seed)
    embeddings = torch.randn(nodes, embedding_dim) * 0.15
    clean_adj = torch.zeros(nodes, nodes)
    for start in range(0, nodes, pieces_per_note):
        end = min(nodes, start + pieces_per_note)
        group_center = torch.randn(embedding_dim)
        for index in range(start, end):
            embeddings[index] += group_center
        for index in range(start, end - 1):
            clean_adj[index, index + 1] = 1.0
            clean_adj[index + 1, index] = 1.0
    clean_adj.fill_diagonal_(0.0)
    return embeddings, clean_adj


def _candidate_incidence(nodes: int, pieces_per_note: int) -> torch.Tensor:
    rows = []
    for start in range(0, nodes, pieces_per_note):
        end = min(nodes, start + pieces_per_note)
        row = torch.zeros(nodes)
        row[start:end] = 1.0
        rows.append(row)
    if nodes >= 4:
        mixed = torch.zeros(nodes)
        mixed[0] = 1.0
        mixed[pieces_per_note if pieces_per_note < nodes else nodes - 1] = 1.0
        rows.append(mixed)
    return torch.stack(rows, dim=0)


def _edge_proxy_metrics(pred_adj: torch.Tensor, clean_adj: torch.Tensor) -> dict[str, float]:
    pred_adj = pred_adj.detach().float().clamp(0.0, 1.0)
    clean_adj = clean_adj.detach().float()
    return {
        "edge_mae": float(torch.mean(torch.abs(pred_adj - clean_adj)).item()),
        "symmetry_error": float(torch.mean(torch.abs(pred_adj - pred_adj.T)).item()),
        "diagonal_error": float(torch.mean(torch.abs(torch.diagonal(pred_adj))).item()),
        "density": float(pred_adj.mean().item()),
    }


def _row(
    architecture: str,
    *,
    elapsed: float,
    metrics: dict[str, float],
    notes: str,
) -> dict[str, Any]:
    proxy_score = (
        metrics.get("edge_mae", 0.0)
        + metrics.get("coverage_loss", 0.0)
        + metrics.get("symmetry_error", 0.0)
        + metrics.get("diagonal_error", 0.0)
    )
    return {
        "architecture": architecture,
        "proxy_score": float(proxy_score),
        "timings_seconds": {"run": elapsed},
        "metrics": metrics,
        "notes": notes,
    }


def _run_v6_smoke(embeddings: torch.Tensor, clean_adj: torch.Tensor) -> dict[str, Any]:
    start = monotonic()
    model = EdgeModel(embedding_dim=embeddings.shape[1])
    gnn = AssemblyGNN(embedding_dim=embeddings.shape[1])
    model.eval()
    gnn.eval()
    with torch.no_grad():
        left = embeddings.unsqueeze(1).repeat(1, embeddings.shape[0], 1).reshape(-1, embeddings.shape[1])
        right = embeddings.unsqueeze(0).repeat(embeddings.shape[0], 1, 1).reshape(-1, embeddings.shape[1])
        pred = model(left, right).reshape(embeddings.shape[0], embeddings.shape[0])
        pred = 0.5 * (pred + pred.T)
        pred.fill_diagonal_(0.0)
        refined = gnn(embeddings, (pred > 0.5).float())
        assignment = sinkhorn_soft_assignment(pred, tau=0.2, iterations=30)
        loss = compute_v6_loss(
            pred.reshape(-1, 1),
            assignment,
            clean_adj.reshape(-1, 1),
            lambda_cover=0.1,
        )
    metrics = _edge_proxy_metrics(pred, clean_adj)
    metrics.update(
        {
            "loss": float(loss.item()),
            "assignment_row_error": float(torch.mean(torch.abs(assignment.sum(dim=1) - 1.0)).item()),
            "refined_embedding_norm": float(refined.norm(dim=1).mean().item()),
        }
    )
    return _row(
        "v6_gnn_soft_cover",
        elapsed=monotonic() - start,
        metrics=metrics,
        notes="Untrained GNN edge scorer plus Sinkhorn pair-assignment smoke pass.",
    )


def _run_v7_smoke(embeddings: torch.Tensor, clean_adj: torch.Tensor, seed: int, steps: int) -> dict[str, Any]:
    start = monotonic()
    np.random.seed(seed)
    model = EnergyNetwork(embedding_dim=embeddings.shape[1])
    model.eval()
    initial = torch.zeros_like(clean_adj)
    with torch.no_grad():
        initial_energy = float(model(embeddings, initial).item())
        pred = mcmc_annealing_search(model, embeddings, initial, steps=steps, temp_start=1.0, temp_end=0.1)
        final_energy = float(model(embeddings, pred).item())
    metrics = _edge_proxy_metrics(pred, clean_adj)
    metrics.update({"initial_energy": initial_energy, "final_energy": final_energy})
    return _row(
        "v7_energy_mcmc",
        elapsed=monotonic() - start,
        metrics=metrics,
        notes="Untrained global energy model with MCMC graph rewiring smoke pass.",
    )


def _run_v8_smoke(embeddings: torch.Tensor, clean_adj: torch.Tensor, seed: int, diffusion_steps: int) -> dict[str, Any]:
    start = monotonic()
    torch.manual_seed(seed)
    model = DenoisingEdgeModel(embedding_dim=embeddings.shape[1])
    diffusion = GraphDiffusionAssembly(model, num_steps=diffusion_steps)
    model.eval()
    with torch.no_grad():
        noisy, noise_mask = diffusion.forward_diffusion(clean_adj, t=max(0, diffusion_steps - 1))
        pred = diffusion.reverse_diffusion(embeddings, noisy)
    metrics = _edge_proxy_metrics(pred, clean_adj)
    metrics.update({"noise_flip_rate": float(noise_mask.float().mean().item())})
    return _row(
        "v8_diffusion",
        elapsed=monotonic() - start,
        metrics=metrics,
        notes="Untrained graph diffusion denoiser conditioned on noisy adjacency.",
    )


def _run_v9_smoke(embeddings: torch.Tensor, pieces_per_note: int) -> dict[str, Any]:
    start = monotonic()
    incidence = _candidate_incidence(embeddings.shape[0], pieces_per_note)
    solver = NeuralILPSolver(embedding_dim=embeddings.shape[1])
    solver.eval()
    candidate_embeddings = incidence @ embeddings / incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
    with torch.no_grad():
        selection_probs = solver(candidate_embeddings).squeeze(-1)
        coverage_loss = candidate_soft_exact_cover_loss(incidence, selection_probs, lambda_entropy=0.0)
        coverage = incidence.T @ selection_probs
    metrics = {
        "coverage_loss": float(coverage_loss.item()),
        "coverage_mae": float(torch.mean(torch.abs(coverage - 1.0)).item()),
        "selection_mean": float(selection_probs.mean().item()),
        "selection_std": float(selection_probs.std(unbiased=False).item()),
        "symmetry_error": 0.0,
        "diagonal_error": 0.0,
    }
    return _row(
        "v9_neural_ilp",
        elapsed=monotonic() - start,
        metrics=metrics,
        notes="Untrained Transformer set-cover selector over candidate assemblies.",
    )


def _run_v10_smoke(embeddings: torch.Tensor, clean_adj: torch.Tensor) -> dict[str, Any]:
    start = monotonic()
    encoder = LatentWorldEncoder(embedding_dim=embeddings.shape[1], latent_dim=max(8, embeddings.shape[1] // 2))
    decoder = LatentWorldDecoder(latent_dim=max(8, embeddings.shape[1] // 2), num_nodes=embeddings.shape[0])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu, _logvar = encoder(embeddings)
        pred, coords = decoder(mu)
        pred = 0.5 * (pred + pred.T)
        pred.fill_diagonal_(0.0)
    metrics = _edge_proxy_metrics(pred, clean_adj)
    metrics.update({"coord_abs_mean": float(torch.abs(coords).mean().item())})
    return _row(
        "v10_latent_world",
        elapsed=monotonic() - start,
        metrics=metrics,
        notes="Untrained latent world encoder/decoder inverse-graphics smoke pass.",
    )


def run_v6_to_v10_architecture_comparison(
    *,
    architectures: Iterable[str] = V6_TO_V10_ARCHITECTURES,
    nodes: int = 8,
    pieces_per_note: int = 4,
    embedding_dim: int = 32,
    seed: int = 7,
    mcmc_steps: int = 20,
    diffusion_steps: int = 6,
) -> dict[str, Any]:
    """Run a deterministic smoke comparison across the v6-v10 model families.

    The returned ``best_architecture`` is a proxy smoke-test winner, not a claim
    of trained reconstruction accuracy. It is useful for checking that every
    modelling route can ingest the same latent graph and emit comparable
    structural metrics before expensive training or real-data evaluation.
    """

    _require_torch()
    requested = tuple(str(item).strip() for item in architectures if str(item).strip())
    unknown = sorted(set(requested) - set(V6_TO_V10_ARCHITECTURES))
    if unknown:
        raise ValueError(f"unknown architectures: {', '.join(unknown)}")

    torch.manual_seed(seed)
    embeddings, clean_adj = _synthetic_clean_graph(nodes, pieces_per_note, embedding_dim, seed)
    rows: list[dict[str, Any]] = []
    for index, architecture in enumerate(requested):
        torch.manual_seed(seed + index * 101)
        if architecture == "v6_gnn_soft_cover":
            rows.append(_run_v6_smoke(embeddings, clean_adj))
        elif architecture == "v7_energy_mcmc":
            rows.append(_run_v7_smoke(embeddings, clean_adj, seed + index * 101, mcmc_steps))
        elif architecture == "v8_diffusion":
            rows.append(_run_v8_smoke(embeddings, clean_adj, seed + index * 101, diffusion_steps))
        elif architecture == "v9_neural_ilp":
            rows.append(_run_v9_smoke(embeddings, pieces_per_note))
        elif architecture == "v10_latent_world":
            rows.append(_run_v10_smoke(embeddings, clean_adj))

    summary = sorted(
        [
            {
                "architecture": row["architecture"],
                "proxy_score": row["proxy_score"],
                "run_seconds": row["timings_seconds"]["run"],
            }
            for row in rows
        ],
        key=lambda item: (item["proxy_score"], item["run_seconds"], item["architecture"]),
    )
    return {
        "config": {
            "architectures": requested,
            "nodes": nodes,
            "pieces_per_note": pieces_per_note,
            "embedding_dim": embedding_dim,
            "seed": seed,
            "mcmc_steps": mcmc_steps,
            "diffusion_steps": diffusion_steps,
            "trained_weights": False,
            "score_interpretation": "lower proxy_score is better for smoke-test structural validity only",
        },
        "rows": rows,
        "summary": summary,
        "best_architecture": summary[0]["architecture"] if summary else None,
    }
