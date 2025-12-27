# ScaleRL Implementation Notes

## Paper References
- ScaleRL (research/papers/arXiv-scalerl/arXiv:2510.13786v1.md)
- Vanilla GRPO (research/papers/arXiv-vanilla_grpo/arXiv:2402.03300.md)
- DAPO (research/papers/arXiv-dapo/arXiv:2503.14476.md)
- RL-ZVP (research/papers/arXiv-rl_zvp/arXiv:2509.21880.md)
- Tricks or Traps? Part I (research/papers/arXiv-tricks_or_traps_part1/arXiv:2508.08221.md)

## Core Defaults
- `LossFunction`: default `CISPOLoss` with asymmetric clipping (`clip_low=0.20`, `clip_high=0.28`), `kl_coeff=0.0`.
- `AdvantageEstimator`: group-mean centering + batch-level std for non-ZVP prompts.
- `alpha_zvp=0.1` for RL-ZVP entropy-scaled advantages; no batch-level rescaling applied in the ZVP branch.
- Reward shaping: asymmetric clipping + truncation masking by default; optional length penalty off (`length_penalty=0.0`).
- Metrics: track `frac_reward_zero_std`, entropy, clipping hit rates, pass-rate statistics, advantage summary stats, KL divergence (logged when active).

## Integration Guardrails
- Always check logits arrive in FP32; raise if precision lower.
- Detect zero-variance prompts using per-group reward std; only apply batch-level std to non-ZVP prompts.
- Upstream sampler will supply prompt groups; assume mixed pass/fail batches and zero-variance prompts retained for RL-ZVP path.
- If additional post-processing scales advantages, ensure RL-ZVP advantages are exempt or scale through `alpha_zvp`.
