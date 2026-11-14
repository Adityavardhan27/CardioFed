"""Unit tests for Integrated Gradients explainability (src/explainability_v2.py).

Verifies mathematical correctness, path integral accuracy, attribution direction,
completeness axiom, and edge cases using analytically tractable models.
Does not require real PTB-XL data or trained checkpoints.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability_v2 import integrated_gradients


class LinearToyModel(nn.Module):
    """Linear model F(x) = Wx + b with shape [B, 12, 1000] -> [B, 5]."""

    def __init__(self, weights, bias=None):
        super().__init__()
        self.weights = nn.Parameter(
            torch.as_tensor(weights, dtype=torch.float32), requires_grad=False
        )
        if bias is not None:
            self.bias = nn.Parameter(
                torch.as_tensor(bias, dtype=torch.float32), requires_grad=False
            )
        else:
            self.bias = None

    def forward(self, x):
        # x: [B, 12, 1000], weights: [5, 12, 1000] -> out: [B, 5]
        out = torch.einsum("blt,clt->bc", x, self.weights)
        if self.bias is not None:
            out = out + self.bias
        return out


class QuadraticToyModel(nn.Module):
    """Quadratic model F_c(x) = sum(w_c * x^2) + b_c."""

    def __init__(self, weights):
        super().__init__()
        self.weights = nn.Parameter(
            torch.as_tensor(weights, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        # x: [B, 12, 1000] -> x^2: [B, 12, 1000]
        return torch.einsum("blt,clt->bc", x ** 2, self.weights)


def test_ig_linear_model_exact_analytical_attribution():
    """For a linear model F(x) = Wx + b, IG attribution is analytically x * W_c."""
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 1, size=(5, 12, 1000)).astype(np.float32)
    bias = np.array([0.5, -0.2, 1.0, -1.5, 0.0], dtype=np.float32)
    model = LinearToyModel(weights, bias)

    x = torch.from_numpy(rng.normal(0, 1, size=(12, 1000)).astype(np.float32))

    for target_class in range(5):
        attr, err, fx, fb = integrated_gradients(model, x, target=target_class, steps=64)

        # Analytical attribution for linear model: (x - baseline) * W = x * W
        expected_attr = x * torch.from_numpy(weights[target_class])

        torch.testing.assert_close(attr, expected_attr, rtol=1e-4, atol=1e-5)
        # Completeness: sum(attr) == fx - fb
        assert err < 1e-4
        assert fx == pytest.approx(float(model(x[None])[0, target_class]), abs=1e-5)
        assert fb == pytest.approx(bias[target_class], abs=1e-5)


def test_ig_attribution_direction_and_sign():
    """Attribution sign must strictly reflect whether a feature increases or decreases the logit."""
    weights = np.zeros((5, 12, 1000), dtype=np.float32)
    # Target class 0:
    # Lead 0: positive weight (+2.0)
    weights[0, 0, :] = 2.0
    # Lead 1: negative weight (-3.0)
    weights[0, 1, :] = -3.0

    model = LinearToyModel(weights)

    # Input:
    # Lead 0, time 0: +1.5 -> expected positive attribution (+1.5 * +2.0 = +3.0)
    # Lead 0, time 1: -1.5 -> expected negative attribution (-1.5 * +2.0 = -3.0)
    # Lead 1, time 0: +1.5 -> expected negative attribution (+1.5 * -3.0 = -4.5)
    # Lead 1, time 1: -1.5 -> expected positive attribution (-1.5 * -3.0 = +4.5)
    x = torch.zeros((12, 1000), dtype=torch.float32)
    x[0, 0] = 1.5
    x[0, 1] = -1.5
    x[1, 0] = 1.5
    x[1, 1] = -1.5

    attr, err, fx, fb = integrated_gradients(model, x, target=0, steps=64)

    assert attr[0, 0].item() > 0.0, "Positive value * positive weight must yield positive attribution"
    assert attr[0, 1].item() < 0.0, "Negative value * positive weight must yield negative attribution"
    assert attr[1, 0].item() < 0.0, "Positive value * negative weight must yield negative attribution"
    assert attr[1, 1].item() > 0.0, "Negative value * negative weight must yield positive attribution"

    # All unweighted leads should have exactly 0 attribution
    assert torch.all(attr[2:, :] == 0.0)


def test_ig_completeness_across_multiple_samples():
    """Verify completeness axiom across multiple distinct random input samples."""
    rng = np.random.default_rng(123)
    weights = rng.normal(0, 0.5, size=(5, 12, 1000)).astype(np.float32)
    bias = rng.normal(0, 1.0, size=(5,)).astype(np.float32)
    model = LinearToyModel(weights, bias)

    for sample_idx in range(5):
        x = torch.from_numpy(rng.normal(0, 1, size=(12, 1000)).astype(np.float32))
        for target_class in range(5):
            attr, err, fx, fb = integrated_gradients(
                model, x, target=target_class, steps=64
            )
            # Completeness: sum(attr) must equal fx - fb
            gap = fx - fb
            abs_err = abs(attr.sum().item() - gap)
            assert abs_err < 1e-4
            assert err < 1e-4


def test_ig_zero_input_produces_zero_attribution():
    """When input equals the baseline (all zeros), all attributions must be zero."""
    rng = np.random.default_rng(999)
    weights = rng.normal(0, 1, size=(5, 12, 1000)).astype(np.float32)
    model = LinearToyModel(weights, bias=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))

    x_zero = torch.zeros((12, 1000), dtype=torch.float32)
    for c in range(5):
        attr, err, fx, fb = integrated_gradients(model, x_zero, target=c, steps=32)
        assert torch.all(attr == 0.0)
        assert fx == pytest.approx(fb, abs=1e-6)
        assert err == 0.0


def test_ig_quadratic_model_midpoint_accuracy():
    """Midpoint Riemann sum is exact for polynomials up to degree 2."""
    rng = np.random.default_rng(777)
    weights = np.ones((5, 12, 1000), dtype=np.float32)
    model = QuadraticToyModel(weights)

    x = torch.from_numpy(rng.uniform(-2, 2, size=(12, 1000)).astype(np.float32))
    attr, err, fx, fb = integrated_gradients(model, x, target=0, steps=64)

    # For F(x) = sum(x^2), baseline=0 -> fb=0, fx = sum(x^2)
    # Path: alpha * x. grad_i = 2 * alpha * x_i
    # Integral_0^1 2 * alpha * x_i d_alpha = x_i
    # Attribution_i = x_i * x_i = x_i^2
    expected_attr = x ** 2
    torch.testing.assert_close(attr, expected_attr, rtol=1e-3, atol=1e-4)
    assert err < 1e-4


def test_app_plot_integrated_gradients_standardized_colormap():
    """Verify that Streamlit plot_integrated_gradients uses the standardized RdBu_r colormap."""
    import matplotlib.pyplot as plt
    from src.app import plot_integrated_gradients

    dummy_signal = torch.zeros((1, 12, 1000), dtype=torch.float32)
    dummy_attr = torch.randn((1, 12, 1000), dtype=torch.float32)

    fig = plot_integrated_gradients(dummy_signal, dummy_attr, "MI", completeness_error=0.01)
    try:
        # axes[1] is the attribution heatmap
        heatmap_ax = fig.axes[1]
        images = heatmap_ax.get_images()
        assert len(images) > 0
        assert images[0].get_cmap().name == "RdBu_r"
    finally:
        plt.close(fig)


def test_app_caption_baseline_disclosure_logic():
    """Verify that STTC and HYP trigger the non-neutral baseline note in Streamlit caption text."""
    import inspect
    from src import app

    src_code = inspect.getsource(app)
    assert 'if explain_class in ["STTC", "HYP"]:' in src_code
    assert "The zero baseline is not neutral for" in src_code
    assert "already-positive baseline prediction" in src_code


def test_app_caption_norm_consistency_disclosure():
    """Verify that when NORM is suppressed by the NORM rule, a disclosure note is added to caption text."""
    import inspect
    from src import app

    src_code = inspect.getsource(app)
    assert 'if explain_class == "NORM" and original[0] == 1 and final[0] == 0:' in src_code
    assert "NORM consistency note:" in src_code
    assert "NORM is suppressed in the final diagnosis" in src_code


def test_ig_device_compatibility():
    """Verify integrated_gradients creates alphas on matching device and works on GPU/MPS if available."""
    weights = np.ones((5, 12, 1000), dtype=np.float32)
    model = LinearToyModel(weights)

    # 1. CPU verification
    x_cpu = torch.randn(12, 1000, dtype=torch.float32)
    attr_cpu, err_cpu, fx_cpu, fb_cpu = integrated_gradients(model, x_cpu, target=0, steps=16)
    assert attr_cpu.device == x_cpu.device
    assert err_cpu < 1e-4

    # 2. Accelerator verification (MPS on Apple Silicon or CUDA on GPU systems)
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if device is not None:
        model_dev = LinearToyModel(weights).to(device)
        x_dev = x_cpu.to(device)
        attr_dev, err_dev, fx_dev, fb_dev = integrated_gradients(model_dev, x_dev, target=0, steps=16)
        assert attr_dev.device == x_dev.device
        assert err_dev < 1e-4
        torch.testing.assert_close(attr_dev.cpu(), attr_cpu, rtol=1e-4, atol=1e-5)


def test_ig_real_model_architectures_and_eval_guard():
    """Verify integrated_gradients runs on actual CardioFed network architectures and enforces eval mode."""
    from src.model import MultiScaleResidualECGNet, ECGNet

    torch.manual_seed(42)
    models = [
        MultiScaleResidualECGNet(n_leads=12, n_classes=5, dropout=0.35),
        ECGNet(n_leads=12, n_classes=5, use_lstm=False, dropout=0.30),
    ]

    x = torch.randn(12, 1000, dtype=torch.float32)

    for model in models:
        # Deliberately place model in training mode to verify integrated_gradients enforces model.eval()
        model.train()
        assert model.training is True

        attr, err, fx, fb = integrated_gradients(model, x, target=1, steps=32, batch=16)

        assert model.training is False, "integrated_gradients must set model.eval()"
        assert attr.shape == (12, 1000)
        assert torch.isfinite(attr).all()
        assert fx == pytest.approx(float(model(x[None])[0, 1].detach()), abs=1e-4)
        # Completeness error on non-linear deep models should be reasonable (< 5%)
        assert err < 0.05


def test_ig_batch_size_invariance():
    """Verify that varying batch chunk size yields identical attributions and completeness."""
    weights = np.random.default_rng(42).normal(0, 1, size=(5, 12, 1000)).astype(np.float32)
    model = LinearToyModel(weights)
    x = torch.randn(12, 1000, dtype=torch.float32)

    attr_16, err_16, _, _ = integrated_gradients(model, x, target=2, steps=64, batch=16)
    attr_32, err_32, _, _ = integrated_gradients(model, x, target=2, steps=64, batch=32)
    attr_64, err_64, _, _ = integrated_gradients(model, x, target=2, steps=64, batch=64)

    torch.testing.assert_close(attr_16, attr_32, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(attr_32, attr_64, rtol=1e-6, atol=1e-6)
    assert err_16 == pytest.approx(err_64, abs=1e-6)


def test_ig_custom_baseline_support():
    """Verify that passing an explicit baseline computes attributions relative to that baseline."""
    weights = np.ones((5, 12, 1000), dtype=np.float32)
    model = LinearToyModel(weights)

    x = torch.ones((12, 1000), dtype=torch.float32) * 2.0
    baseline = torch.ones((12, 1000), dtype=torch.float32) * 0.5

    attr, err, fx, fb = integrated_gradients(
        model, x, target=0, steps=32, baseline=baseline
    )

    # For linear model F(x) = sum(x):
    # fx = 12 * 1000 * 2.0 = 24000
    # fb = 12 * 1000 * 0.5 = 6000
    # expected attr per element = (2.0 - 0.5) * 1.0 = 1.5
    expected_attr = (x - baseline) * 1.0
    torch.testing.assert_close(attr, expected_attr, rtol=1e-4, atol=1e-4)
    assert fx == pytest.approx(24000.0, abs=1.0)
    assert fb == pytest.approx(6000.0, abs=1.0)
    assert err < 1e-4


def test_app_integrated_gradients_wrapper():
    """Verify that src.app.integrated_gradients correctly wraps ig_v2_core and handles 3D batch dimension."""
    from src.app import integrated_gradients as app_ig

    weights = np.ones((5, 12, 1000), dtype=np.float32)
    model = LinearToyModel(weights)

    # App passes 3D signal [1, 12, 1000]
    signal = torch.randn(1, 12, 1000, dtype=torch.float32)
    attr, err = app_ig(model, signal, target_class=1, steps=32)

    assert attr.shape == (1, 12, 1000)
    assert isinstance(err, float)
    assert err < 1e-4

    # Direct call to core function on 2D slice
    attr_core, err_core, _, _ = integrated_gradients(model, signal[0], target=1, steps=32)
    torch.testing.assert_close(attr[0], attr_core, rtol=1e-6, atol=1e-6)
    assert err == pytest.approx(err_core, abs=1e-6)


def test_deletion_scores_monotonicity_and_random_control():
    """Verify that deletion_scores causes a larger logit drop for top attributed features than random features."""
    from src.explainability_v2 import deletion_scores

    # Linear model where feature (0, 0) has massive weight 100.0, others have weight 0.1
    weights = np.ones((5, 12, 1000), dtype=np.float32) * 0.1
    weights[0, 0, 0] = 100.0
    model = LinearToyModel(weights)

    x = torch.ones((12, 1000), dtype=torch.float32)
    attr, _, _, _ = integrated_gradients(model, x, target=0, steps=32)

    rng = np.random.default_rng(42)
    scores = deletion_scores(model, x, attr, target=0, rng=rng, n_random=10)

    assert len(scores) == 4
    for entry in scores:
        assert entry["drop_top"] > entry["drop_random"], (
            f"Fraction {entry['fraction']}: drop_top ({entry['drop_top']}) must exceed "
            f"drop_random ({entry['drop_random']})"
        )


def test_ig_invalid_target_class_bounds():
    """Verify that requesting an out-of-bounds target class raises IndexError."""
    weights = np.ones((5, 12, 1000), dtype=np.float32)
    model = LinearToyModel(weights)
    x = torch.randn(12, 1000, dtype=torch.float32)

    with pytest.raises(IndexError):
        integrated_gradients(model, x, target=10, steps=16)

    with pytest.raises(IndexError):
        integrated_gradients(model, x, target=-10, steps=16)


