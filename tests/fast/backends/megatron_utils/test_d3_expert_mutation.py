import types

import pytest
import torch

from miles.backends.megatron_utils.actor import MegatronTrainRayActor


class _TensorBackuperNormal:
    def __init__(self, weights):
        self.weights = weights

    def get(self, key):
        assert key == "actor"
        return self.weights


def _fake_actor(*, colocate=True, asleep=True):
    weights = {
        "decoder.layers.0.mlp.experts.linear_fc1.weight": torch.zeros(4, dtype=torch.bfloat16),
        "decoder.layers.0.self_attention.linear_qkv.weight": torch.zeros(4),
    }
    fake = types.SimpleNamespace(
        args=types.SimpleNamespace(
            colocate=colocate, debug_disable_optimizer=True, num_rollout=0
        ),
        _asleep=asleep,
        _weight_sync_reads_tms_backup=False,
        weights_backuper=_TensorBackuperNormal(weights),
    )
    fake._get_actor_weights = lambda: fake.weights_backuper.get("actor")
    return fake, weights


def test_d3_mutation_modifies_authoritative_cpu_source(monkeypatch):
    monkeypatch.setattr(
        "miles.backends.megatron_utils.actor.dist.get_rank", lambda: 0
    )
    fake, weights = _fake_actor()
    result = MegatronTrainRayActor.debug_d3_mutate_expert_source.__wrapped__(fake)
    assert result["rank"] == 0
    assert result["before"] == 0.0
    assert result["after"] == 1.0
    assert float(weights[result["tensor"]].flatten()[0]) == 1.0


@pytest.mark.parametrize("colocate,asleep", [(False, True), (True, False)])
def test_d3_mutation_rejects_non_smoke_state(colocate, asleep):
    fake, weights = _fake_actor(colocate=colocate, asleep=asleep)
    with pytest.raises(RuntimeError, match="requires idle"):
        MegatronTrainRayActor.debug_d3_mutate_expert_source.__wrapped__(fake)
    assert float(next(iter(weights.values())).flatten()[0]) == 0.0
