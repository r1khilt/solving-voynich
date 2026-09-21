import json

import torch

from voynich.communication.coupling import score_action_candidates
from voynich.communication.dynamics import ActionWorldModel, DynamicsConfig
from voynich.communication.pipeline import make_episode, verify_candidate
from voynich.communication.schema import JointLayout


def test_compiled_procedure_is_scored_without_reordering_or_mutation():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        layout = JointLayout(24, 128)
        episode = make_episode(2, "validation", layout, family="procedure")
        hypothesis = layout.unpack(episode.clean, episode.observation)
        report = {
            "candidates": [
                {
                    "id": "chosen-before-score",
                    "hypothesis": hypothesis,
                    "verification": verify_candidate(episode.observation, hypothesis),
                }
            ]
        }
        before = json.dumps(report)
        model = ActionWorldModel(DynamicsConfig(width=16, layers=1, heads=2))
        model.train()
        result = score_action_candidates(report, model)
        assert model.training
        assert json.dumps(report) == before
        candidate = result["candidates"][0]
        assert candidate["id"] == "chosen-before-score"
        assert candidate["action_model"]["status"] == "scored"
        assert candidate["action_model"]["steps"] == episode.world.config.events
        assert 0 <= candidate["action_model"]["mean_applicability_probability"] <= 1
        json.dumps(result, allow_nan=False)
    finally:
        torch.set_num_threads(old)
