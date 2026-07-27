"""Generation is chunked because the prefill logits tensor is the VRAM ceiling.

`LlamaForCausalLM.forward` materialises [batch, seq, vocab] logits and then calls
`logits.float()`. At Llama-3's 128k vocab a 100-prompt batch at seq~77 is a single 3.68 GiB
fp32 allocation, which OOMs a 14.5 GiB T4 already holding half the sharded weights — the exact
failure that killed the first `asw eval` on Kaggle's 2xT4. `device_map="auto"` cannot help: it
splits weights across devices, not one tensor.

Two things must hold, and only one of them needs a GPU:
  * the chunk size actually reaches every generator (torch-free — the config-threading tests),
  * chunking does not change the output (torch-guarded — runs on the GPU host).
"""
import pytest

from asw.harness.cli import _batch_size, _build_generator
from asw.harness.generate import chunk_bounds

try:
    import torch
except Exception:                      # ImportError or broken DLLs on the authoring box
    torch = None

# a MARKER, not a module-level skip: the config-threading tests below need no torch and must
# still run on the authoring box, where they are the only guard against a silent mismatch
requires_torch = pytest.mark.skipif(torch is None, reason="torch unavailable on this host")


def _cfg(tmp, batch_size=None):
    cfg = {"paths": {"cache_dir": str(tmp)},
           "model": {"id": "x/y", "steer_layers": [13, 14, 15, 16]}}
    if batch_size is not None:
        cfg["decoding"] = {"batch_size": batch_size}
    return cfg


# ── config threading (no torch needed) ────────────────────────────────────────
def test_batch_size_defaults_when_config_is_silent(tmp_path):
    assert _batch_size(_cfg(tmp_path)) == 16
    assert _batch_size({}) == 16


def test_batch_size_reaches_the_undefended_generator(tmp_path):
    assert _build_generator(_cfg(tmp_path, 4), None, None, "none", 8.0).batch_size == 4


def test_batch_size_reaches_the_system_prompt_defense(tmp_path):
    assert _build_generator(_cfg(tmp_path, 4), None, None, "system_prompt", 8.0).batch_size == 4


def test_batch_size_reaches_the_steered_defenses(tmp_path):
    """A wrapper that quietly kept batch=16 while the config said 4 would OOM exactly where the
    undefended arm survived, making the defense look broken for a memory reason."""
    import numpy as np

    from asw.geometry.extract import save_drefuse

    d = {l: np.ones(8) for l in (13, 14, 15, 16)}
    save_drefuse(tmp_path / "drefuse" / "x_y.npz", d)
    gen = _build_generator(_cfg(tmp_path, 4), None, None, "abliteration", 8.0)
    assert gen.batch_size == 4


def test_wrapper_passes_its_batch_size_to_its_own_generator(tmp_path):
    """Wrapper.generate builds an HFGenerator internally; that one must inherit the bound."""
    import numpy as np

    from asw.wrapper.wrapper import Wrapper

    d = {13: np.ones(8)}
    w = Wrapper(None, None, d, {13: "raw_add"}, 8.0, batch_size=4)
    assert w.batch_size == 4


# ── chunk/mask alignment (no torch needed) ────────────────────────────────────
# The regression this guards: generation was chunked while the wrapper's per-row condition mask
# stayed whole-batch, so the steering hook met 16 hidden states and a 100-long mask and died with
# "The size of tensor a (16) must match the size of tensor b (100)". `cast` and `wrapper` are
# gated and hit it; `abliteration` passes mask=None and broadcast, which is why it survived.
def test_chunk_bounds_cover_every_index_exactly_once():
    for n, step in [(100, 16), (10, 3), (7, 7), (5, 8), (1, 16), (16, 16)]:
        bounds = chunk_bounds(n, step)
        covered = [i for lo, hi in bounds for i in range(lo, hi)]
        assert covered == list(range(n)), (n, step)


def test_chunk_bounds_edge_cases():
    assert chunk_bounds(0, 16) == []            # empty prompt list -> no batches at all
    assert chunk_bounds(5, None) == [(0, 5)]    # batch_size=None -> one batch
    assert chunk_bounds(100, 16)[-1] == (96, 100)


def test_mask_slice_always_matches_its_prompt_chunk():
    """THE invariant. A mask slice shorter or longer than its batch is the crash."""
    for n, step in [(100, 16), (10, 3), (7, 7), (5, 8), (1, 16)]:
        prompts = list(range(n))
        mask = [i % 2 == 0 for i in range(n)]
        for lo, hi in chunk_bounds(n, step):
            assert len(mask[lo:hi]) == len(prompts[lo:hi]) > 0, (n, step, lo, hi)


def test_mask_slices_keep_each_prompt_with_its_own_flag():
    """Alignment, not just length: row i of a batch must carry prompt i's decision."""
    n, step = 100, 16
    mask = [i % 3 == 0 for i in range(n)]
    rebuilt = [flag for lo, hi in chunk_bounds(n, step) for flag in mask[lo:hi]]
    assert rebuilt == mask


# ── behaviour (needs torch; runs on the GPU host) ─────────────────────────────
class _Tok:
    """Minimal tokenizer: one token per character, left-padded like the real loader configures."""

    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[-1]["content"]

    def __call__(self, texts, return_tensors=None, padding=True, add_special_tokens=False):
        width = max(len(t) for t in texts)
        ids = [[0] * (width - len(t)) + [ord(c) for c in t] for t in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), width,
                                                                             dtype=torch.long)}

    def batch_decode(self, gen, skip_special_tokens=True):
        return ["".join(chr(int(t)) for t in row if int(t) != 0) for row in gen]


class _Model:
    """Echoes each prompt's last character, and records the batch size of every call."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.calls.append(int(input_ids.shape[0]))
        last = input_ids[:, -1:]
        return torch.cat([input_ids, last], dim=1)


def _gen(batch_size):
    from asw.harness.generate import HFGenerator

    return HFGenerator(_Model(), _Tok(), batch_size=batch_size)


@requires_torch
def test_chunking_splits_into_the_expected_number_of_calls():
    g = _gen(4)
    g.generate([f"prompt{i}" for i in range(10)], temperature=0.0, max_new_tokens=1, seed=0)
    assert g.model.calls == [4, 4, 2]


@requires_torch
def test_chunked_output_matches_unchunked_exactly():
    """The whole point: this is a memory fix, so it must be behaviour-preserving.

    Prompt lengths deliberately vary (1..10 chars) so each chunk pads to a different width than
    the single-batch run would. Left-padding plus the attention mask is what makes that
    irrelevant to the result — if chunking ever changed padding semantics, this catches it."""
    prompts = [f"{'x' * i}{chr(97 + i)}" for i in range(10)]
    kw = dict(temperature=0.0, max_new_tokens=1, seed=0)
    one_shot = _gen(None).generate(prompts, **kw)
    chunked = _gen(3).generate(prompts, **kw)
    assert chunked == one_shot
    assert len(chunked) == len(prompts)          # order and count preserved across chunks


@requires_torch
def test_batch_size_none_sends_one_call():
    g = _gen(None)
    g.generate([f"p{i}" for i in range(7)], temperature=0.0, max_new_tokens=1, seed=0)
    assert g.model.calls == [7]


@requires_torch
def test_empty_prompt_list_does_not_call_the_model():
    g = _gen(4)
    assert g.generate([], temperature=0.0, max_new_tokens=1, seed=0) == []
    assert g.model.calls == []


@requires_torch
def test_gated_wrapper_generates_in_chunks_without_a_mask_shape_error(monkeypatch):
    """End-to-end reproduction of the crash: a gated wrapper over more prompts than one batch.

    Before the fix this raised "The size of tensor a (16) must match the size of tensor b (100)"
    from the steering hook, because the mask covered all prompts while `h` was one chunk."""
    import numpy as np

    from asw.wrapper.wrapper import Wrapper

    n, bs = 100, 16
    prompts = [f"{'x' * (i % 5)}p{i}" for i in range(n)]

    # _mask does `from ..geometry.extract import capture_terminal` INSIDE the method, so the
    # patch has to land on the source module — patching the wrapper's namespace would be a no-op.
    monkeypatch.setattr(
        "asw.geometry.extract.capture_terminal",
        lambda model, tok, ps, layers, **kw: {layers[0]: np.arange(len(ps), dtype=float)[:, None]})

    class _Cond:                     # flags alternating rows, so the mask is genuinely mixed
        def predict(self, acts):
            return np.asarray(acts).ravel() % 2 == 0

    seen = []

    class _RecordingSteer:
        """Stands in for WrapperSteer to record the mask each chunk actually receives."""

        def __init__(self, model, d_by_layer, branch_by_layer, alpha, mask=None, site="block"):
            seen.append(None if mask is None else len(mask))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("asw.wrapper.steer.WrapperSteer", _RecordingSteer)

    w = Wrapper(_Model(), _Tok(), {13: np.ones(4)}, {13: "raw_add"}, 8.0,
                condition=_Cond(), condition_layer=13, batch_size=bs)
    outs = w.generate(prompts, temperature=0.0, max_new_tokens=1, seed=0)

    assert len(outs) == n
    # every chunk's mask matches that chunk's batch — the shape error was 100 against 16
    assert seen == [16] * 6 + [4]
    assert w.model.calls == [16] * 6 + [4]
