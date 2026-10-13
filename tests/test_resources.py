"""CPU-only checks for bundled data; no checkpoints or text models required."""

import numpy as np
import pytest
import torch
from hymotion.pipeline import motion_diffusion
from hymotion.pipeline.body_model import WoodenMesh
from hymotion.resources import asset_path


def test_assets_work_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("Mean.npy", "Std.npy"):
        values = np.load(asset_path(f"stats/{name}"), allow_pickle=False)
        assert values.shape == (201,)
        assert np.isfinite(values).all()
    mesh = WoodenMesh()
    assert mesh.num_joints == 52
    assert mesh.num_verts > 0


@pytest.mark.parametrize("mean_std_dir", [None, "stats", "./stats"])
def test_pipeline_uses_bundled_statistics(tmp_path, monkeypatch, mean_std_dir):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(motion_diffusion, "load_object", lambda *args, **kwargs: torch.nn.Identity())
    pipeline = motion_diffusion.MotionGeneration(
        network_module="unused",
        network_module_args={},
        text_encoder_module="unused",
        text_encoder_cfg={},
        mean_std_dir=mean_std_dir,
    )
    assert pipeline.body_model.num_joints == 52
    assert pipeline.mean.shape == (201,)
    assert pipeline.std.shape == (201,)
    np.testing.assert_array_equal(
        pipeline.mean.numpy(), np.load(asset_path("stats/Mean.npy")).astype(np.float32)
    )
    np.testing.assert_array_equal(
        pipeline.std.numpy(), np.load(asset_path("stats/Std.npy")).astype(np.float32)
    )


def test_missing_asset_fails():
    with pytest.raises(FileNotFoundError, match="runtime asset is missing"):
        asset_path("stats/not-an-asset.npy")
