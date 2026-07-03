"""The geometry-aware conditional wrapper (C4), as a drop-in Generator.

Flow per `generate` call:
  1. condition pre-pass: capture the condition-layer terminal residual for each prompt and
     ask the ConditionVector which are harmful -> a per-row mask (no condition => steer all).
  2. generate under WrapperSteer with that mask, so benign rows pass through untouched
     (XSTest/utility protected) and harmful rows get the geometry-appropriate operator.

Because it satisfies the Generator interface, `evaluate_benchmark`, the baselines harness, and
the attack runner all work on the wrapper with no special-casing.
"""
from __future__ import annotations

import numpy as np


class Wrapper:
    def __init__(self, model, tok, d_by_layer, branch_by_layer, alpha, *,
                 condition=None, condition_layer=None, site="block", system_prompt=None):
        self.model = model
        self.tok = tok
        self.d_by_layer = {int(k): v for k, v in d_by_layer.items()}
        self.branch_by_layer = {int(k): v for k, v in branch_by_layer.items()}
        self.alpha = alpha
        self.condition = condition
        self.condition_layer = condition_layer
        self.site = site
        self.system_prompt = system_prompt

    def _mask(self, prompts):
        """Per-row harmful flag from the condition vector; None => steer every row."""
        if self.condition is None:
            return None
        import torch

        from ..geometry.extract import capture_terminal

        acts = capture_terminal(self.model, self.tok, prompts, [self.condition_layer],
                                assistant=None)[self.condition_layer]
        flags = self.condition.predict(acts)            # numpy bool [N]
        return torch.as_tensor(np.asarray(flags), dtype=torch.bool)

    def steer_context(self, mask=None):
        """The wrapper's steering hooks as a standalone context manager (WrapperSteer).

        Used by the adaptive attack (Item 6, Step 2): passing this as `steer=` to run_gcg makes
        the geometry branch active across every forward/backward pass, so GCG optimises the suffix
        against the DEFENDED model — the honest "through-the-defense" attack. `mask=None` steers
        every row (i.e. assume the detector fired), which is the strongest defensive posture."""
        from .steer import WrapperSteer

        return WrapperSteer(self.model, self.d_by_layer, self.branch_by_layer,
                            self.alpha, mask=mask, site=self.site)

    def generate(self, prompts, *, temperature, max_new_tokens, seed):
        from ..harness.generate import HFGenerator
        from .steer import WrapperSteer

        mask = self._mask(prompts)
        gen = HFGenerator(self.model, self.tok, system_prompt=self.system_prompt)
        with WrapperSteer(self.model, self.d_by_layer, self.branch_by_layer,
                          self.alpha, mask=mask, site=self.site):
            return gen.generate(prompts, temperature=temperature,
                                max_new_tokens=max_new_tokens, seed=seed)

    @classmethod
    def from_geometry_map(cls, model, tok, d_by_layer, anti_alignment_map, alpha,
                          *, force_op=None, **kw):
        """Build branch assignments from a per-layer anti-alignment map (classify_geometry).

        `force_op` in {"raw_add", "project"} overrides the geometry-chosen operator on every band
        layer; running the wrapper forced to each operator on both aligned and anti-aligned models
        yields the clean operator x geometry conditions for the crossover interaction (Item 4)."""
        from .steer import branch_for_label

        keys = {int(k) for k in d_by_layer}
        branch = {int(l): (force_op or branch_for_label(v["label"]))
                  for l, v in anti_alignment_map.items() if int(l) in keys}
        return cls(model, tok, d_by_layer, branch, alpha, **kw)
