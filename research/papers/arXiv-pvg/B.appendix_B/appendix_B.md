# PROVER-VERIFIER GAMES IMPROVE LEGIBILITY OF LLM OUTPUTS
Jan Hendrik Kirchner\* Yining Chen\*
Harri Edwards† Jan Leike† Nat McAleese Yuri Burda†
OpenAI

## B SEPARATION MIGHT INHIBIT PROGRESSION OF CHECKABILITY TRAINING
A verifier trained to predict correctness of solutions might not in fact learn a generalizable algorithm for predicting correctness, if other attributes of the solutions are highly correlated with correctness and are easier to predict.
In our experiments, all training rounds of the verifier, except the very first one, contain correct and incorrect solutions produced by helpful and sneaky provers. Since our training incentivizes helpful prover's solutions to be correct, and sneaky prover's solutions to be incorrect, correctness of solutions in the verifier's training data strongly correlates with which of the two provers produced that solution. If the helpful and sneaky provers produce solutions in different easily identifiable formats, and helpful solutions are mostly correct, while sneaky solutions are mostly incorrect, then training the verifier on the correctness signal will likely generalize to identify distinctive format features of the solutions produced by the helpful and sneaky provers.
Figure S3 presents evidence that in the early rounds of checkability training, helpful and sneaky provers are in fact easily separable based on easy to learn features, but grow less separable as the training progresses. For that figure, we trained a very small discriminator to predict whether a given sample was produced by the helpful or the sneaky prover on a small balanced dataset of helpful and sneaky solutions. We trained separate discriminators for checkpoints of the prover spaced every 400 optimization steps of training. The figure shows ROC AUC of that discriminator. In the first round, ROC AUC reaches close to 1 at the end of the round. For the last round, it stays close to 0.5 for the whole duration of the round.
![](cb74fd9f5ec715dd3e2e325b864b48bc_img.jpg)
### a SRC checkability training
| RL steps | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
|----------|---------|---------|---------|---------|---------|
| 0k       | 0.60    | 0.60    | 0.60    | 0.60    | 0.60    |
| 1k       | 0.80    | 0.75    | 0.65    | 0.62    | 0.60    |
| 2k       | 0.90    | 0.85    | 0.70    | 0.65    | 0.60    |
| 3k       | 0.95    | 0.75    | 0.65    | 0.68    | 0.60    |
| 4k       | 0.98    | 0.85    | 0.75    | 0.62    | 0.60    |
| 5k       | 1.00    | 0.90    | 0.65    | 0.63    | 0.60    |
| 6k       | 1.00    | 0.90    | 0.65    | 0.63    | 0.60    |
| 7k       | 1.00    | 0.90    | 0.65    | 0.63    | 0.60    |
### b CGC checkability training
| RL steps | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
|----------|---------|---------|---------|---------|---------|
| 0k       | 0.60    | 0.60    | 0.60    | 0.60    | 0.60    |
| 1k       | 0.95    | 0.90    | 0.85    | 0.80    | 0.75    |
| 2k       | 1.00    | 0.98    | 0.95    | 0.90    | 0.85    |
| 3k       | 1.00    | 0.98    | 0.95    | 0.90    | 0.80    |
| 4k       | 1.00    | 1.00    | 0.98    | 0.95    | 0.85    |
| 5k       | 1.00    | 1.00    | 1.00    | 0.98    | 0.90    |
Supplementary Figure S3: **Evolution of separability of helpful and sneaky provers over the course of training.** **a** When training with SRC reward, in early rounds the provers become easily separable after a short number of optimization steps. In later rounds of training it's difficult for a small discriminator to tell helpful samples apart from sneaky ones. **b** When training with CGC reward, the helpful and sneaky provers remain highly separable throughout training.
We hypothesize that samples produced by the helpful and sneaky provers after they separate completely in a way that is easily learnable by very small verifiers (i.e. based on mostly surface level statistics of the solutions) are not useful for increasing the robustness of the verifier. As a consequence, when designing particular formulations of a Prover-Verifier Game, one has to pay attention to whether it in fact encourages early separation of the helpful and sneaky players.
