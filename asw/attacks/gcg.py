"""GCG adversarial suffix attack (Zou et al. 2023). Tier-1 (A100), query-budgeted.

Optimizes an adversarial suffix appended to a harmful instruction to maximize the probability
of an affirmative target completion ("Sure, here is ..."), then checks whether the *steered*
target actually complies. Host-run torch; pure helpers are unit-tested.

NOTE: validate this implementation against nanoGCG / llm-attacks on a known-jailbreakable
case before trusting absolute ASR numbers — token-boundary handling under chat templates is
the usual source of silent breakage (see scripts/validate_gcg.py, Step 5). The relative
comparison (wrapper vs naive vs SP) is the claim, and all three are attacked identically.

The suffix is optimised INSIDE the user turn, at the exact slot the attack driver fills at
eval time (`behavior + " " + suffix`): a sentinel marks that slot in the chat-templated
string and the string is split on it, so the optimised tokens transfer to generation. The
templated text is tokenised with add_special_tokens=False to match HFGenerator/capture_terminal
(no double BOS) — the detector-aware penalty reads the same terminal token the wrapper does.
"""
from __future__ import annotations

from dataclasses import dataclass

AFFIRMATIVE = "Sure, here is"
# Sentinel marking where the adversarial suffix is spliced into the chat-templated prompt.
# Plain ASCII so no tokenizer treats it as a special token; removed by the split before use.
_OPTIM_SLOT = "[[GCG-OPTIM-SLOT]]"


@dataclass
class GCGConfig:
    n_steps: int = 250
    search_width: int = 256          # candidate substitutions evaluated per step
    topk: int = 256                  # top-k tokens per position by -gradient
    suffix_len: int = 20
    init_suffix: str = " ".join(["!"] * 20)
    target_prefix: str = AFFIRMATIVE
    batch_size: int = 64             # candidate eval batch (memory bound)


def target_string(behavior_target: str | None = None, prefix: str = AFFIRMATIVE) -> str:
    """The completion we optimize the model to begin with."""
    return behavior_target if behavior_target else f"{prefix} "


def condition_penalty(proj_cond, tau, margin: float = 0.0, lam: float = 1.0) -> float:
    """Detector-aware hinge  lam * max(0, proj_cond - tau + margin): push the condition-vector
    projection below the wrapper's firing threshold tau so the defense never engages. Pure
    reference for the differentiable torch term used inside run_gcg."""
    return float(lam) * max(0.0, float(proj_cond) - float(tau) + float(margin))


def run_gcg(model, tok, instruction, target=None, config: GCGConfig | None = None, *,
            steer=None, condition=None, condition_layer=None,
            tau: float = 0.0, penalty_lambda: float = 0.0, penalty_margin: float = 0.0):
    """Optimize an adversarial suffix. Returns (best_suffix, best_loss, n_queries, history).

    Standard GCG: gradient w.r.t. one-hot suffix tokens -> top-k candidates -> sampled
    substitutions -> keep the lowest-loss suffix, repeated for n_steps. ADAPTIVE variants (Item 6):

      steer=<context manager>   attack the DEFENDED model: the wrapper/steering hooks are active
                                across every forward/backward (h + alpha*v is differentiable, so
                                gradients flow through the defense). Static ASR against a steered
                                model is not a robustness result under current norms; this is.
      condition=<ConditionVector>, condition_layer, penalty_lambda>0
                                detector-aware attack: add lam*relu(proj_cond - tau + margin) to
                                the loss so the suffix slips past the harmful-input detector (use
                                with steer=None -- evade detection so the real wrapper never fires).

    `history` is a list of (step, best_loss, queries) for ASR-vs-budget curves.
    """
    import contextlib

    import torch
    import torch.nn.functional as F

    cfg = config or GCGConfig()
    device = model.device
    embed_w = model.get_input_embeddings().weight
    tgt = target_string(target)

    def ids(text, add_special=False):
        return tok(text, return_tensors="pt", add_special_tokens=add_special).input_ids[0].to(device)

    # Splice the suffix into the user turn at the driver's eval-time slot. `add_generation_prompt`
    # closes the user turn and opens the assistant header, so post_str carries that header and the
    # suffix stays where the adversary can actually control it.
    templated = tok.apply_chat_template(
        [{"role": "user", "content": instruction + " " + _OPTIM_SLOT}],
        tokenize=False, add_generation_prompt=True)
    pre_str, post_str = templated.split(_OPTIM_SLOT)
    pre_ids = ids(pre_str)                  # user turn up to the suffix (BOS matched to HFGenerator)
    post_ids = ids(post_str)                # rest of the user turn + assistant generation header
    suffix_ids = ids(cfg.init_suffix)
    target_ids = ids(tgt)

    cond_dir = None
    if condition is not None and penalty_lambda > 0:
        from ..models.hooks import get_module, hidden_of
        cond_dir = torch.as_tensor(getattr(condition, "direction", condition),
                                   device=device, dtype=embed_w.dtype)
        cond_dir = cond_dir / cond_dir.norm()
        cond_pos = -(target_ids.shape[0] + 1)   # terminal prompt token the detector reads

    queries = 0
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()
    history: list = []

    def build(sfx):
        return torch.cat([pre_ids, sfx, post_ids, target_ids])

    def loss_of(seqs):
        # Candidate ranking uses the SAME objective the gradient step optimises: affirmative-target
        # cross-entropy plus (detector-aware) the evasion hinge. Ranking on CE alone would let the
        # gradient point toward evasion while selection ignored it, so the suffix would never
        # actually drop the detector projection — the penalty must gate selection too.
        cap2, h2 = {}, None
        if cond_dir is not None:
            def _ch(_m, _i, out, _c=cap2):
                _c["h"] = hidden_of(out)[:, cond_pos, :]
            h2 = get_module(model, condition_layer).register_forward_hook(_ch)
        out = model(seqs).logits
        tstart = seqs.shape[1] - target_ids.shape[0]
        logits = out[:, tstart - 1:-1, :]
        tgt_rep = target_ids.unsqueeze(0).expand(seqs.shape[0], -1)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_rep.reshape(-1),
                             reduction="none").view(seqs.shape[0], -1).mean(1)
        if cond_dir is not None:
            ce = ce + penalty_lambda * torch.relu((cap2["h"] @ cond_dir) - tau + penalty_margin)
            h2.remove()
        return ce

    sl = slice(pre_ids.shape[0], pre_ids.shape[0] + suffix_ids.shape[0])
    ts = slice(-target_ids.shape[0], None)

    with (steer if steer is not None else contextlib.nullcontext()):
        for step in range(cfg.n_steps):
            seq = build(suffix_ids)
            one_hot = torch.zeros(suffix_ids.shape[0], embed_w.shape[0],
                                  device=device, dtype=embed_w.dtype)
            one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0).requires_grad_()
            embeds = model.get_input_embeddings()(seq.unsqueeze(0)).detach()
            embeds[0, sl] = one_hot @ embed_w

            cap, handle = {}, None
            if cond_dir is not None:
                def _chook(_m, _i, out, _cap=cap):
                    _cap["h"] = hidden_of(out)[0, cond_pos, :]      # kept in the graph
                handle = get_module(model, condition_layer).register_forward_hook(_chook)

            logits = model(inputs_embeds=embeds).logits
            loss = F.cross_entropy(logits[0, ts.start - 1:-1, :], target_ids)
            if cond_dir is not None:
                loss = loss + penalty_lambda * torch.relu((cap["h"] @ cond_dir) - tau + penalty_margin)
                handle.remove()
            loss.backward()
            grad = one_hot.grad

            topk = (-grad).topk(cfg.topk, dim=1).indices            # [suffix_len, topk]
            cands = suffix_ids.repeat(cfg.search_width, 1)
            pos = torch.randint(0, suffix_ids.shape[0], (cfg.search_width,), device=device)
            choice = torch.randint(0, cfg.topk, (cfg.search_width,), device=device)
            cands[torch.arange(cfg.search_width), pos] = topk[pos, choice]

            with torch.no_grad():
                seqs = torch.stack([build(c) for c in cands])
                losses = torch.cat([loss_of(seqs[i:i + cfg.batch_size])
                                    for i in range(0, seqs.shape[0], cfg.batch_size)])
                queries += seqs.shape[0]
                j = int(losses.argmin())
                if float(losses[j]) < best_loss:
                    best_loss = float(losses[j])
                    suffix_ids = cands[j].clone()
                    best_suffix = suffix_ids.clone()
            history.append((step, best_loss, queries))

    return tok.decode(best_suffix), best_loss, queries, history
