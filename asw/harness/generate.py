"""Batched generation with resume.

A `Generator` is anything with `.generate(prompts, *, temperature, max_new_tokens, seed)
-> list[str]`. `HFGenerator` is the real one; `ScriptedGenerator` lets the harness be
unit-tested without a GPU.

`run_generation` is resumable at prompt granularity: it reads any existing parquet,
reuses finished `prompt_id`s, and only generates the remainder — so a Kaggle 12h cutoff
or a RunPod spot reclaim never recomputes a finished prompt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from ..data.benchmarks import Example


class Generator(Protocol):
    def generate(self, prompts, *, temperature: float, max_new_tokens: int, seed: int): ...


class ScriptedGenerator:
    """Test/debug generator: `fn(prompt) -> response`."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def generate(self, prompts, *, temperature, max_new_tokens, seed):
        return [self.fn(p) for p in prompts]


class HFGenerator:
    """Real generator. Applies the chat template and (optionally) steering hooks.

    `hooks` is a list of forward-hook handles already registered on the model (e.g. the
    wrapper's steering hooks); generation simply runs with them attached. Loading and
    hook registration live in asw/models (Steps 3/8); this class only drives decoding.
    """

    def __init__(self, model, tokenizer, *, system_prompt: str | None = None):
        self.model = model
        self.tok = tokenizer
        self.system_prompt = system_prompt

    def _format(self, prompt: str) -> str:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, *, temperature, max_new_tokens, seed):
        import torch
        from .. import repro

        repro.set_seed(seed)
        texts = [self._format(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        do_sample = temperature and temperature > 0.0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


def run_generation(
    generator: Generator,
    examples: list[Example],
    *,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    resume_path: str | Path | None = None,
) -> list[dict]:
    """Generate one response per example; reuse finished prompt_ids from resume_path."""
    done: dict = {}
    if resume_path and Path(resume_path).exists():
        import pandas as pd

        prev = pd.read_parquet(resume_path)
        prev = prev[prev["temperature"] == temperature]
        done = dict(zip(prev["prompt_id"], prev["response"]))

    todo = [e for e in examples if e.id not in done]
    fresh = (
        generator.generate(
            [e.prompt for e in todo],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if todo
        else []
    )
    produced = dict(zip([e.id for e in todo], fresh))

    rows: list[dict] = []
    for e in examples:
        rows.append(
            {
                "prompt_id": e.id,
                "category": e.category,
                "prompt": e.prompt,
                "response": done.get(e.id, produced.get(e.id)),
                "temperature": temperature,
                "seed": seed,
                "reused": e.id in done,
            }
        )
    return rows
