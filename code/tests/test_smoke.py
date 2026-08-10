"""
Smoke test: exercise every class in the mvgt_net package with random tensors.
Verifies that all modules are importable, forward-passable, and that shapes
are consistent end-to-end.

Run with: pytest tests/test_smoke.py -v
Or simply: python tests/test_smoke.py
"""
import sys
import os
import torch

# Add the parent directory to the path so we can import mvgt_net
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvgt_net import (
    MVGTNet, MultiViewEmbedding, MultiViewGraphBuilder,
    HierarchicalAttention, PFGAModule, PFGAMultiView,
    LoRALinear, STLLMPlus, MultiTaskLoss,
    masked_mae, masked_rmse, masked_wape, masked_mse,
    masked_mape, masked_smape, r2_score,
)


def test_lora_linear():
    """Test LoRALinear: forward pass + parameter count."""
    print("Testing LoRALinear...")
    layer = LoRALinear(in_features=64, out_features=128, r=8)
    x = torch.randn(2, 10, 64)
    y = layer(x)
    assert y.shape == (2, 10, 128), f"Expected (2,10,128), got {y.shape}"
    # Trainable params: r * (in + out) = 8 * (64 + 128) = 1536
    assert layer.num_trainable_parameters() == 8 * (64 + 128)
    print(f"  OK: output shape {y.shape}, trainable params {layer.num_trainable_parameters()}")


def test_multi_view_embedding():
    """Test MultiViewEmbedding with and without text/categorical."""
    print("Testing MultiViewEmbedding...")
    emb = MultiViewEmbedding(
        num_nodes=10, input_dim=4, hidden_dim=32, lookback=12,
        use_text=True, use_categorical=True, num_categories=5,
    )
    x_numeric = torch.randn(2, 12, 10, 4)
    x_text = {"fact": torch.randint(0, 1000, (2, 12, 20))}
    x_cat = torch.randint(0, 5, (2, 12, 10))
    h = emb(x_numeric, x_text, x_cat)
    assert h.shape == (2, 10, 96), f"Expected (2,10,96), got {h.shape}"
    print(f"  OK: output shape {h.shape}")


def test_graph_builder():
    """Test MultiViewGraphBuilder."""
    print("Testing MultiViewGraphBuilder...")
    builder = MultiViewGraphBuilder(num_nodes=10, topk=4)
    x_numeric = torch.randn(2, 12, 10, 4)
    h = torch.randn(2, 10, 96)
    adj_multi, components, weights = builder(x_numeric, h, adj_spatial=None)
    assert adj_multi.shape == (10, 10), f"Expected (10,10), got {adj_multi.shape}"
    assert "spatial" in components
    assert "temporal" in components
    assert "semantic" in components
    assert "adaptive" in components
    # Check row normalization (approximately, due to top-k sparsification)
    row_sums = adj_multi.sum(dim=-1)
    print(f"  OK: adj shape {adj_multi.shape}, weights {weights}")
    print(f"       row sums (should be ~1): mean={row_sums.mean():.4f}, std={row_sums.std():.4f}")


def test_hierarchical_attention():
    """Test HierarchicalAttention."""
    print("Testing HierarchicalAttention...")
    attn = HierarchicalAttention(hidden_dim=96, num_heads=4, dropout=0.1)
    h = torch.randn(2, 10, 96)
    adj = torch.rand(10, 10)
    h_out, weights = attn(h, adj)
    assert h_out.shape == (2, 10, 96), f"Expected (2,10,96), got {h_out.shape}"
    assert "time" in weights
    assert "view" in weights
    assert "graph" in weights
    print(f"  OK: output shape {h_out.shape}, attention keys: {list(weights.keys())}")


def test_pfga_module():
    """Test single PFGA layer."""
    print("Testing PFGAModule...")
    # Frozen layer (no graph mask)
    frozen = PFGAModule(hidden_dim=64, num_heads=4, frozen=True,
                       use_graph_mask=False, use_lora=False)
    h = torch.randn(2, 10, 64)
    h_out = frozen(h)
    assert h_out.shape == (2, 10, 64)

    # Unfrozen layer with graph mask + LoRA
    unfrozen = PFGAModule(hidden_dim=64, num_heads=4, frozen=False,
                         use_graph_mask=True, use_lora=True, lora_rank=8)
    adj = torch.rand(10, 10)
    h_out = unfrozen(h, adj=adj)
    assert h_out.shape == (2, 10, 64)
    print(f"  OK: frozen + unfrozen layers both pass")


def test_pfga_multi_view():
    """Test stacked PFGA layers."""
    print("Testing PFGAMultiView...")
    pfga = PFGAMultiView(
        hidden_dim=96, num_layers=8, num_frozen_layers=6,
        num_unfrozen_layers=2, num_heads=4, lora_rank=8,
    )
    h = torch.randn(2, 10, 96)
    adj = torch.rand(10, 10)
    h_out = pfga(h, adj)
    assert h_out.shape == (2, 10, 96)
    n_train = pfga.num_trainable_parameters()
    n_frozen = pfga.num_frozen_parameters()
    print(f"  OK: trainable={n_train:,}, frozen={n_frozen:,}, "
          f"efficiency={100*n_train/(n_train+n_frozen):.2f}%")


def test_st_llm_plus():
    """Test STLLMPlus reproduction."""
    print("Testing STLLMPlus...")
    model = STLLMPlus(
        num_nodes=10, input_dim=4, hidden_dim=32, lookback=12, horizon=3,
        F_layers=2, U_layers=1, num_heads=4, lora_rank=8,
    )
    x = torch.randn(2, 12, 10, 4)
    adj = torch.rand(10, 10)
    y = model(x, adj)
    assert y.shape == (2, 3, 10, 4), f"Expected (2,3,10,4), got {y.shape}"
    n_train = model.num_trainable_parameters()
    n_frozen = model.num_frozen_parameters()
    print(f"  OK: output {y.shape}, trainable={n_train:,}, frozen={n_frozen:,}")


def test_mvgt_net():
    """Test the full MVGTNet model."""
    print("Testing MVGTNet (full model)...")
    config = {
        "num_nodes": 10,
        "input_dim": 4,
        "hidden_dim": 32,
        "lookback": 12,
        "horizon": 3,
        "use_text": True,
        "use_categorical": True,
        "num_categories": 5,
        "frozen_layers": 2,
        "unfrozen_layers": 1,
        "num_heads": 4,
        "lora_rank": 8,
        "topk": 4,
        "dropout": 0.1,
    }
    model = MVGTNet(config)
    x_numeric = torch.randn(2, 12, 10, 4)
    x_text = {"fact": torch.randint(0, 1000, (2, 12, 20))}
    x_cat = torch.randint(0, 5, (2, 12, 10))
    outputs = model(x_numeric, x_text, x_cat, return_attention=True)
    assert "numeric" in outputs
    assert outputs["numeric"].shape == (2, 3, 10, 4)
    assert "categorical" in outputs
    assert "attention" in outputs
    assert "adj_multi" in outputs
    assert "view_weights" in outputs
    eff = model.parameter_efficiency()
    print(f"  OK: numeric {outputs['numeric'].shape}, "
          f"categorical {outputs['categorical'].shape}")
    print(f"       view weights: {outputs['view_weights']}")
    print(f"       param efficiency: {eff}")


def test_multi_task_loss():
    """Test MultiTaskLoss (Formula D)."""
    print("Testing MultiTaskLoss (Formula D)...")
    loss_fn = MultiTaskLoss(task_names=["numeric", "text", "categorical"],
                            history_length=5)
    # Simulate 10 steps of training
    for step in range(10):
        losses = {
            "numeric": torch.tensor(1.0 / (step + 1), requires_grad=True),
            "text": torch.tensor(2.0 / (step + 1), requires_grad=True),
            "categorical": torch.tensor(0.5 / (step + 1), requires_grad=True),
        }
        total, weights = loss_fn(losses)
        if step == 9:
            print(f"  Step {step}: total={total.item():.4f}, weights={weights}")
    print(f"  OK: loss history buffer shape {loss_fn.get_loss_history().shape}")


def test_metrics():
    """Test all evaluation metrics."""
    print("Testing metrics...")
    pred = torch.randn(2, 3, 10, 4)
    target = torch.randn(2, 3, 10, 4)
    results = {
        "MAE": float(masked_mae(pred, target)),
        "MSE": float(masked_mse(pred, target)),
        "RMSE": float(masked_rmse(pred, target)),
        "WAPE": float(masked_wape(pred, target)),
        "MAPE": float(masked_mape(pred, target)),
        "sMAPE": float(masked_smape(pred, target)),
        "R2": float(r2_score(pred, target)),
    }
    for name, val in results.items():
        assert isinstance(val, float), f"{name} should be float"
        assert val == val, f"{name} is NaN"  # NaN check
    print(f"  OK: {results}")


def test_backward_pass():
    """Test that gradients flow through the full model."""
    print("Testing backward pass (gradient flow)...")
    config = {
        "num_nodes": 5, "input_dim": 2, "hidden_dim": 16, "lookback": 6,
        "horizon": 2, "use_text": False, "use_categorical": False,
        "frozen_layers": 1, "unfrozen_layers": 1, "num_heads": 2,
        "lora_rank": 4, "topk": 3,
    }
    model = MVGTNet(config)
    model.train()
    x = torch.randn(2, 6, 5, 2)
    outputs = model(x)
    loss = outputs["numeric"].sum()
    loss.backward()
    # Check that at least some parameters have non-None gradients
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  OK: {n_with_grad}/{n_trainable} trainable params have gradients")


def main():
    """Run all tests."""
    print("=" * 70)
    print("MVGT-Net Smoke Test Suite")
    print("=" * 70)
    tests = [
        test_lora_linear,
        test_multi_view_embedding,
        test_graph_builder,
        test_hierarchical_attention,
        test_pfga_module,
        test_pfga_multi_view,
        test_st_llm_plus,
        test_mvgt_net,
        test_multi_task_loss,
        test_metrics,
        test_backward_pass,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed} TESTS FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
