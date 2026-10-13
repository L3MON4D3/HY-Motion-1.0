"""Exercise real ODE solvers without weights or text encoders."""

from itertools import pairwise

import pytest
import torch
from hymotion.pipeline import motion_diffusion


@pytest.fixture(params=["cpu", "cuda"])
def pipeline(monkeypatch, request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, **kwargs):
            self.calls += 1
            return 0.05 * x

    denoiser = Denoiser()
    monkeypatch.setattr(motion_diffusion, "load_object", lambda *args, **kwargs: denoiser)
    result = motion_diffusion.MotionFlowMatching(
        network_module="unused",
        network_module_args={"input_dim": 201, "vtxt_input_dim": 2, "ctxt_input_dim": 2},
        text_encoder_module="unused",
        text_encoder_cfg={},
        mean_std_dir="stats",
        infer_noise_scheduler_cfg={"validation_steps": 4},
        train_frames=30,
        random_generator_on_gpu=False,
    )
    result.to(device)
    monkeypatch.setattr(result, "encode_text", lambda _: {
        "text_vec_raw": torch.zeros(1, 1, 2, device=device),
        "text_ctxt_raw": torch.zeros(1, 1, 2, device=device),
        "text_ctxt_raw_length": torch.tensor([1], device=device),
    })
    monkeypatch.setattr(result, "decode_motion_from_latent", lambda sample, **kwargs: {"latent": sample.clone()})
    return result


@pytest.mark.parametrize("method,evaluations_per_step", [("euler", 1), ("rk4", 4)])
def test_progress_counts_completed_steps_not_model_evaluations(pipeline, method, evaluations_per_step):
    pipeline._noise_scheduler_cfg = {"method": method}
    updates = []

    def report(value):
        updates.append((value, pipeline.motion_transformer.calls))

    output = pipeline.generate("wave", [42], 1, progress_callback=report)
    assert [value for value, _ in updates] == pytest.approx([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    assert [calls for _, calls in updates] == [
        0, 0, evaluations_per_step, 2 * evaluations_per_step,
        3 * evaluations_per_step, 4 * evaluations_per_step, 4 * evaluations_per_step,
    ]
    # A callback must not change the solver, random stream, or output.
    baseline = pipeline.generate("wave", [42], 1)
    assert torch.equal(output["latent"], baseline["latent"])


def test_default_fifty_step_sampling_reports_every_completed_step(pipeline):
    pipeline.validation_steps = 50
    updates = []
    pipeline.generate("wave", [42], 1, progress_callback=updates.append)
    assert updates == pytest.approx(
        [0, 0.1] + [0.1 + 0.8 * step / 50 for step in range(1, 50)] + [0.9, 1]
    )
    assert pipeline.motion_transformer.calls == 50


def test_adaptive_solver_reports_monotonic_completed_time(pipeline):
    pipeline._noise_scheduler_cfg = {"method": "dopri5"}
    updates = []
    output = pipeline.generate("wave", [42], 1, progress_callback=updates.append)
    assert updates[:2] == [0, 0.1]
    assert updates[-2:] == [0.9, 1]
    assert all(a < b for a, b in pairwise(updates))
    assert torch.equal(output["latent"], pipeline.generate("wave", [42], 1)["latent"])


def test_callback_exception_stops_sampling_and_no_callback_leaks(pipeline):
    class Cancelled(Exception):
        pass

    def cancel(value):
        if value >= 0.3:
            raise Cancelled

    with pytest.raises(Cancelled):
        pipeline.generate("wave", [42], 1, progress_callback=cancel)
    assert pipeline.motion_transformer.calls == 1
    # The next job gets a fresh callback and succeeds on the cached pipeline.
    updates = []
    pipeline.generate("wave", [42], 1, progress_callback=updates.append)
    assert updates[-1] == 1


def test_completion_is_not_reported_when_decoding_fails(pipeline, monkeypatch):
    updates = []

    def fail(*args, **kwargs):
        raise RuntimeError("decoding failed")

    monkeypatch.setattr(pipeline, "decode_motion_from_latent", fail)
    with pytest.raises(RuntimeError, match="decoding failed"):
        pipeline.generate("wave", [42], 1, progress_callback=updates.append)
    assert updates[-1] == 0.9


def test_scipy_solver_rejects_unsupported_progress(pipeline):
    pipeline._noise_scheduler_cfg = {"method": "scipy_solver"}
    with pytest.raises(ValueError, match="native torchdiffeq"):
        pipeline.generate("wave", [42], 1, progress_callback=lambda _: None)
