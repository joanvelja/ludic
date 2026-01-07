# PROVER-VERIFIER GAMES IMPROVE LEGIBILITY OF LLM OUTPUTS
Jan Hendrik Kirchner\* Yining Chen\*
Harri Edwards† Jan Leike† Nat McAleese Yuri Burda†
OpenAI

## E VALIDATION WITH PROVERS WITH NO MATH-SPECIFIC PRETRAINING.
![Two line graphs showing accuracy versus fraction completed for 'helpful' and 'sneaky' tasks across 6 rounds of training (init., 0, 1, 2, 3, 4). The 'helpful' graph shows accuracy generally stable between 0.3 and 0.6. The 'sneaky' graph shows a sharp decline in accuracy to near zero as the fraction completed approaches 1.0, especially in earlier rounds.](19f5ea8afbf380781eb0645831a88987_img.jpg)
| Type    | Round | Fraction: 0.0 | Fraction: 0.2 | Fraction: 0.4 | Fraction: 0.6 | Fraction: 0.8 | Fraction: 1.0 |
|---------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| helpful | init. | 0.40          | 0.50          | 0.50          | 0.50          | 0.45          | 0.35          |
| helpful | 0     | 0.40          | 0.55          | 0.55          | 0.50          | 0.45          | 0.30          |
| helpful | 1     | 0.40          | 0.50          | 0.55          | 0.50          | 0.45          | 0.35          |
| helpful | 2     | 0.40          | 0.55          | 0.55          | 0.50          | 0.45          | 0.35          |
| helpful | 3     | 0.40          | 0.55          | 0.60          | 0.55          | 0.50          | 0.40          |
| helpful | 4     | 0.40          | 0.55          | 0.60          | 0.55          | 0.50          | 0.40          |
| sneaky  | init. | 0.40          | 0.50          | 0.50          | 0.40          | 0.20          | 0.05          |
| sneaky  | 0     | 0.40          | 0.20          | 0.10          | 0.05          | 0.05          | 0.05          |
| sneaky  | 1     | 0.40          | 0.45          | 0.40          | 0.10          | 0.05          | 0.05          |
| sneaky  | 2     | 0.40          | 0.50          | 0.55          | 0.45          | 0.20          | 0.05          |
| sneaky  | 3     | 0.40          | 0.50          | 0.55          | 0.45          | 0.25          | 0.05          |
| sneaky  | 4     | 0.40          | 0.50          | 0.55          | 0.45          | 0.30          | 0.05          |
Two line graphs showing accuracy versus fraction completed for 'helpful' and 'sneaky' tasks across 6 rounds of training (init., 0, 1, 2, 3, 4). The 'helpful' graph shows accuracy generally stable between 0.3 and 0.6. The 'sneaky' graph shows a sharp decline in accuracy to near zero as the fraction completed approaches 1.0, especially in earlier rounds.
Supplementary Figure S9: **When repeating our proposed checkability training with LLMs that have not been finetuned on math-specific text, the dynamics remain qualitatively unchanged.**