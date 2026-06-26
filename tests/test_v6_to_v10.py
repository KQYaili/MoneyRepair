"""Tests for the v6 to v10 advanced modeling architectures in MoneyRepair."""

from __future__ import annotations

import pytest

# Skip all tests in this file if torch is not installed
torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F
from moneyrepair.v6_to_v10 import (
    FragmentEncoder,
    GraphAttentionLayer,
    EdgeModel,
    AssemblyGNN,
    V6_TO_V10_ARCHITECTURES,
    V6TrainingSmokeConfig,
    candidate_soft_exact_cover_loss,
    make_assembly_training_sample,
    sinkhorn_soft_assignment,
    compute_v6_loss,
    EnergyNetwork,
    mcmc_annealing_search,
    DenoisingEdgeModel,
    GraphDiffusionAssembly,
    NeuralILPSolver,
    LatentWorldEncoder,
    LatentWorldDecoder,
    run_v6_to_v10_architecture_comparison,
    run_v6_training_smoke,
)


def test_v6_gnas_forward_and_loss():
    # 1. Test FragmentEncoder
    encoder = FragmentEncoder(embedding_dim=128)
    dummy_images = torch.randn(4, 3, 64, 64)
    dummy_masks = torch.randn(4, 1, 64, 64)
    embeddings = encoder(dummy_images, dummy_masks)
    assert embeddings.shape == (4, 128)

    # 2. Test EdgeModel
    edge_model = EdgeModel(embedding_dim=128)
    e_i = embeddings[0:2]
    e_j = embeddings[2:4]
    p_match = edge_model(e_i, e_j)
    assert p_match.shape == (2, 1)

    # 3. Test GraphAttentionLayer and AssemblyGNN
    gnn = AssemblyGNN(embedding_dim=128)
    adj = torch.ones(4, 4)
    refined_embeddings = gnn(embeddings, adj)
    assert refined_embeddings.shape == (4, 128)

    # 4. Test Sinkhorn soft assignment
    P = torch.rand(4, 4)
    A = sinkhorn_soft_assignment(P, tau=0.1, iterations=50)
    assert A.shape == (4, 4)
    assert torch.allclose(A.sum(dim=1), torch.ones(4), atol=1e-3)
    assert torch.allclose(A.sum(dim=0), torch.ones(4), atol=1e-3)

    # 5. Test loss computation
    y_edge = torch.tensor([[1.0], [0.0]])
    P_match_loss = torch.tensor([[0.9], [0.1]])
    loss = compute_v6_loss(P_match_loss, A, y_edge)
    assert loss.item() > 0.0

    # 6. Candidate-level soft exact cover: candidate selections cover fragments exactly once
    incidence = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    good_selection = torch.tensor([1.0, 1.0, 0.0])
    bad_selection = torch.tensor([1.0, 0.0, 1.0])
    good_loss = candidate_soft_exact_cover_loss(incidence, good_selection, lambda_entropy=0.0)
    bad_loss = candidate_soft_exact_cover_loss(incidence, bad_selection, lambda_entropy=0.0)
    assert good_loss < bad_loss


def test_v7_ebm_forward_and_mcmc():
    energy_net = EnergyNetwork(embedding_dim=64)
    node_embeddings = torch.randn(5, 64)
    adj = torch.zeros(5, 5)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0

    energy = energy_net(node_embeddings, adj)
    assert energy.shape == ()
    
    # Test MCMC simulated annealing search
    optimized_adj = mcmc_annealing_search(
        energy_net,
        node_embeddings,
        adj,
        steps=10,
        temp_start=1.0,
        temp_end=0.1,
    )
    assert optimized_adj.shape == (5, 5)
    assert (optimized_adj == 0.0).all() or (optimized_adj == 1.0).all() or optimized_adj.sum() > 0


def test_v8_diffusion_forward_and_reverse():
    model = DenoisingEdgeModel(embedding_dim=64)
    diffusion = GraphDiffusionAssembly(model, beta_start=0.001, beta_end=0.02, num_steps=5)
    
    node_embeddings = torch.randn(6, 64)
    clean_adj = torch.zeros(6, 6)
    clean_adj[0, 1] = 1.0
    clean_adj[1, 0] = 1.0
    
    # Forward diffusion corruption
    noisy_adj, noise_mask = diffusion.forward_diffusion(clean_adj, t=3)
    assert noisy_adj.shape == (6, 6)
    assert noise_mask.shape == (6, 6)
    assert torch.allclose(noisy_adj, noisy_adj.T)
    assert torch.allclose(noisy_adj.diagonal(), torch.zeros(6))
    
    # Reverse diffusion denoising
    random_noise_adj = torch.randint(0, 2, (6, 6)).float()
    random_noise_adj = 0.5 * (random_noise_adj + random_noise_adj.T)
    random_noise_adj.fill_diagonal_(0.0)
    denoised_adj = diffusion.reverse_diffusion(node_embeddings, random_noise_adj)
    assert denoised_adj.shape == (6, 6)
    assert torch.allclose(denoised_adj, denoised_adj.T)
    assert torch.allclose(denoised_adj.diagonal(), torch.zeros(6))


def test_v9_neural_ilp_solver():
    solver = NeuralILPSolver(embedding_dim=64)
    candidate_embeddings = torch.randn(8, 64) # 8 candidate assemblies
    selection_probs = solver(candidate_embeddings)
    assert selection_probs.shape == (8, 1)
    assert (selection_probs >= 0.0).all() and (selection_probs <= 1.0).all()


def test_v10_latent_world_model():
    encoder = LatentWorldEncoder(embedding_dim=64, latent_dim=32)
    decoder = LatentWorldDecoder(latent_dim=32, num_nodes=10)
    
    fragment_embeddings = torch.randn(10, 64)
    mu, logvar = encoder(fragment_embeddings)
    assert mu.shape == (1, 32)
    assert logvar.shape == (1, 32)
    
    # Reparameterization trick
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    
    reconstructed_adj, coordinates = decoder(z)
    assert reconstructed_adj.shape == (10, 10)
    assert coordinates.shape == (10, 2)
    assert (reconstructed_adj >= 0.0).all() and (reconstructed_adj <= 1.0).all()


def test_v6_to_v10_architecture_comparison_runs_all_routes():
    payload = run_v6_to_v10_architecture_comparison(
        nodes=6,
        pieces_per_note=3,
        embedding_dim=16,
        seed=13,
        mcmc_steps=5,
        diffusion_steps=3,
    )

    assert payload["config"]["trained_weights"] is False
    assert payload["best_architecture"] in V6_TO_V10_ARCHITECTURES
    assert {row["architecture"] for row in payload["rows"]} == set(V6_TO_V10_ARCHITECTURES)
    assert len(payload["summary"]) == len(V6_TO_V10_ARCHITECTURES)
    assert all("proxy_score" in row for row in payload["summary"])


def test_v6_training_smoke_generates_hard_negatives_and_diagnostics():
    sample = make_assembly_training_sample(
        nodes=6,
        pieces_per_note=3,
        embedding_dim=16,
        seed=19,
        hard_negative_top_k=3,
    )
    assert sample.node_embeddings.shape == (6, 16)
    assert sample.edge_labels.shape == (6, 6)
    assert sample.candidate_fragment_incidence.shape[1] == 6
    assert len(sample.hard_negative_edges) == 3

    payload = run_v6_training_smoke(
        V6TrainingSmokeConfig(
            nodes=6,
            pieces_per_note=3,
            embedding_dim=16,
            seed=19,
            hard_negative_top_k=3,
            steps=6,
            lr=0.02,
        )
    )

    assert len(payload["loss_curve"]) == 7
    assert payload["final"]["loss"] < payload["initial"]["loss"]
    assert payload["collapse_diagnostics"]["final"]["edge_entropy"] >= 0.0
    assert payload["final"]["hard_negative_edges"]
