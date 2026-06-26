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
from typing import Any, Tuple

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
