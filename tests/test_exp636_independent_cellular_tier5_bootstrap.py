import copy

import pytest
import yaml

from experiments.exp_636_independent_cellular_tier5_bootstrap.run import validate_stages


def config():
    with open("configs/exp_636_independent_cellular_tier5_bootstrap.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_frozen_curriculum_has_declared_order_and_work():
    candidate = config()
    validate_stages(candidate)
    assert sum(stage["steps"] for stage in candidate["training"]["stages"]) == 3200
    assert candidate["training"]["stages"][-1]["name"] == "recovery"


def test_stage_guard_rejects_reduced_training_work():
    candidate = copy.deepcopy(config())
    candidate["training"]["stages"][0]["steps"] -= 1
    with pytest.raises(ValueError, match="3,200"):
        validate_stages(candidate)
