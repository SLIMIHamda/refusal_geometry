"""GCG adversarial suffix attack (Zou et al. 2023). Tier-1 (A100), query-budgeted.

Optimizes an adversarial suffix appended to a harmful instruction to maximize the probability
of an affirmative target completion ("Sure, here is ..."), then checks whether the *steered*
target actually complies. Host-run torch; pure helpers are unit-tested.

NOTE: validate this implementation against nanoGCG / llm-attacks on a known-jailbreakable
case before trusting absolute ASR numbers — token-boundary handling under chat templates is
the usual source of silent breakage. The relative comparison (wrapper vs naive vs SP) is the
claim, and all three are attacked identically.
"""
from __future__ import annotations

from dataclasses import dataclass

AFFIRMATIVE = "Sure, here is"


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


def run_gcg(model, tok, instruction, target=None, config: GCGConfig | None = None):
    """Optimize an adversarial suffix. Returns (best_suffix, best_loss, n_queries).

    Host-run (torch). Standard GCG: gradient w.r.t. one-hot suffix tokens -> top-k candidates
    -> sampled substitutions -> keep the lowest-loss suffix, repeated for n_steps.
    """
    import torch
    import torch.nn.functional as F

    cfg = config or GCGConfig()
    device = model.device
    embed_w = model.get_input_embeddings().weight
    tgt = target_string(target)

    def ids(text, add_special=False):
        return tok(text, return_tensors="pt", add_special_tokens=add_special).input_ids[0].to(device)

    prefix = tok.apply_chat_template([{"role": "user", "content": instruction + " "}],
                                     tokenize=False, add_generation_prompt=True)
    prefix_ids = ids(prefix, add_special=True)
    suffix_ids = ids(cfg.init_suffix)
    target_ids = ids(tgt)

    queries = 0
    best_loss = float("inf")
    best_suffix = suffix_ids.clone()

    def build(sfx):
        return torch.cat([prefix_ids, sfx, target_ids])

    def loss_of(seqs):
        # seqs: [B, L] ; CE on the target span only
        out = model(seqs).logits
        tstart = seqs.shape[1] - target_ids.shape[0]
        logits = out[:, tstart - 1:-1, :]
        tgt_rep = target_ids.unsqueeze(0).expand(seqs.shape[0], -1)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_rep.reshape(-1),
                               reduction="none").view(seqs.shape[0], -1).mean(1)

    sl = slice(prefix_ids.shape[0], prefix_ids.shape[0] + suffix_ids.shape[0])
    ts = slice(-target_ids.shape[0], None)

    for _ in range(cfg.n_steps):
        seq = build(suffix_ids)
        one_hot = torch.zeros(suffix_ids.shape[0], embed_w.shape[0],
                              device=device, dtype=embed_w.dtype)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0).requires_grad_()
        embeds = model.get_input_embeddings()(seq.unsqueeze(0)).detach()
        embeds[0, sl] = one_hot @ embed_w
        logits = model(inputs_embeds=embeds).logits
        loss = F.cross_entropy(logits[0, ts.start - 1:-1, :], target_ids)
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

    return tok.decode(best_suffix), best_loss, queries
