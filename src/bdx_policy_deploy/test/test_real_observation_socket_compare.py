import json

import numpy as np
import pytest

from bdx_policy_deploy.policy_interface import OBS_DIM
from bdx_policy_deploy.real_observation_socket_compare_node import (
    parse_real_observation_payload,
    summarize_observation_difference,
)


def test_parse_real_observation_json_list() -> None:
    values = [float(index) for index in range(OBS_DIM)]

    obs = parse_real_observation_payload(json.dumps(values).encode("utf-8"))

    np.testing.assert_allclose(obs, values)


def test_parse_real_observation_json_object_obs() -> None:
    values = [float(index) * 0.1 for index in range(OBS_DIM)]

    obs = parse_real_observation_payload(json.dumps({"obs": values}).encode("utf-8"))

    np.testing.assert_allclose(obs, values)


def test_parse_real_observation_binary_float32() -> None:
    values = np.arange(OBS_DIM, dtype=np.float32)

    obs = parse_real_observation_payload(values.astype("<f4").tobytes())

    np.testing.assert_allclose(obs, values)


def test_parse_real_observation_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        parse_real_observation_payload(json.dumps({"obs": [0.0]}).encode("utf-8"))


def test_summarize_observation_difference() -> None:
    real = np.ones(OBS_DIM, dtype=np.float32)
    sim = np.zeros(OBS_DIM, dtype=np.float32)

    summary = summarize_observation_difference(real, sim)

    assert summary["max_abs"] == pytest.approx(1.0)
    assert summary["rms"] == pytest.approx(1.0)
    assert summary["blocks"]["joint_pos"]["max_abs"] == pytest.approx(1.0)
