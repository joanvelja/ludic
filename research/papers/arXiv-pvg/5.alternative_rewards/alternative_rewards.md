# PROVER-VERIFIER GAMES IMPROVE LEGIBILITY OF LLM OUTPUTS
Jan Hendrik Kirchner\* Yining Chen\*
Harri Edwards† Jan Leike† Nat McAleese Yuri Burda†
OpenAI

## 5 ALTERNATIVE REWARDS
We now investigate whether using the alternative rewards for prover training affects convergence as well as legibility.
![Figure 9: Dynamics of the prover training phase with the CGC reward across rounds 0, 1, 2, and 5. Top row: Average training verifier score (y-axis: -2 to 8). Bottom row: Training prover accuracy (y-axis: 0 to 1). X-axis: RL steps (0 to 8k). Legend: Helpful (red), Sneaky (grey), Correct (solid), Incorrect (dotted). In the verifier score plots, helpful correct and sneaky correct scores generally increase, while sneaky incorrect scores rise to meet helpful correct scores. In the accuracy plots, helpful accuracy increases to ~0.8, while sneaky accuracy drops to ~0.](7c1f9e78e0f033d391b687f1652f6e47_img.jpg)
| Round | Metric         | Prover Type | Correctness | Initial Value (0 steps) | Final Value (approx. 8k steps) |
|-------|----------------|-------------|-------------|-------------------------|--------------------------------|
| 0     | Verifier Score | Helpful     | Correct     | ~2                      | ~4                             |
| 0     | Verifier Score | Helpful     | Incorrect   | ~0                      | ~2                             |
| 0     | Verifier Score | Sneaky      | Correct     | ~2                      | ~4                             |
| 0     | Verifier Score | Sneaky      | Incorrect   | ~0                      | ~2                             |
| 0     | Accuracy       | Helpful     | N/A         | ~0.4                    | ~0.75                          |
| 0     | Accuracy       | Sneaky      | N/A         | ~0.4                    | ~0                             |
| 1     | Verifier Score | Helpful     | Correct     | ~-1                     | ~5                             |
| 1     | Verifier Score | Helpful     | Incorrect   | ~-1                     | ~2.5                           |
| 1     | Verifier Score | Sneaky      | Correct     | ~-1                     | ~4.5                           |
| 1     | Verifier Score | Sneaky      | Incorrect   | ~-1                     | ~4.5                           |
| 1     | Accuracy       | Helpful     | N/A         | ~0.4                    | ~0.8                           |
| 1     | Accuracy       | Sneaky      | N/A         | ~0.5                    | ~0                             |
| 2     | Verifier Score | Helpful     | Correct     | ~0                      | ~5                             |
| 2     | Verifier Score | Helpful     | Incorrect   | ~0                      | ~3                             |
| 2     | Verifier Score | Sneaky      | Correct     | ~0                      | ~4.5                           |
| 2     | Verifier Score | Sneaky      | Incorrect   | ~0                      | ~4.5                           |
| 2     | Accuracy       | Helpful     | N/A         | ~0.4                    | ~0.8                           |
| 2     | Accuracy       | Sneaky      | N/A         | ~0.5                    | ~0                             |
| 5     | Verifier Score | Helpful     | Correct     | ~3                      | ~6                             |
| 5     | Verifier Score | Helpful     | Incorrect   | ~3                      | ~4                             |
| 5     | Verifier Score | Sneaky      | Correct     | ~3                      | ~5.5                           |
| 5     | Verifier Score | Sneaky      | Incorrect   | ~3                      | ~6                             |
| 5     | Accuracy       | Helpful     | N/A         | ~0.4                    | ~0.8                           |
| 5     | Accuracy       | Sneaky      | N/A         | ~0.5                    | ~0                             |
Figure 9: Dynamics of the prover training phase with the CGC reward across rounds 0, 1, 2, and 5. Top row: Average training verifier score (y-axis: -2 to 8). Bottom row: Training prover accuracy (y-axis: 0 to 1). X-axis: RL steps (0 to 8k). Legend: Helpful (red), Sneaky (grey), Correct (solid), Incorrect (dotted). In the verifier score plots, helpful correct and sneaky correct scores generally increase, while sneaky incorrect scores rise to meet helpful correct scores. In the accuracy plots, helpful accuracy increases to ~0.8, while sneaky accuracy drops to ~0.
Figure 9: **Dynamics of the prover training phase with the CGC reward across rounds.** The average training verifier score (top), and the training prover accuracy for the helpful and sneaky provers (bottom). The verifier is  $10^{-3}$  times the size of the prover in terms of pretraining compute. The number of RL steps until sneaky incorrect become as convincing as helpful correct decreases from round 1 to round 3 and increases from round 3 to round 5.
### 5.1 CHECKABILITY TRAINING WITH THE CORRECTNESS-GATED CONVINCINGNESS (CGC) REWARD
The Correctness-Gated Convincingness (CGC) reward (3) prioritizes correctness (for the helpful prover) and incorrectness (for the sneaky prover) over convincingness. When the helpful prover produces incorrect answers (or the sneaky prover produces correct answers), the reward is set to a constant penalty; otherwise it is the verifier score. In this section, we show that the CGC reward empirically leads to worse training dynamics.
In each round of CGC reward training, we train the prover until the average verifier score of the sneaky’s incorrect samples become as high as that of the helpful’s correct samples, and the sneaky reaches low ( $< 5\%$ ) accuracy. Figure 9 shows the prover training dynamics — the number of RL steps until the sneaky prover succeeds does not consistently increase over rounds. We observe that the sneaky prover’s accuracy quickly decreases and the convincingness rises more slowly for the CGC reward in comparison to the SRC reward. Our hypothesis is that in this case, the sneaky and helpful players quickly exploits different modes with distinct styles, and hence there are obvious spurious statistical clues that separates sneaky and helpful which won’t help the verifier learn the correctness signal. See section C for more discussion. It is possible that we did not tune the hyperparameters for the CGC reward optimally, or the verifier requires more rounds to become robust. We did not perform human evaluation for the CGC reward training due to its worse training dynamics.
Nevertheless, this alternative reward formulation serves as a strong attack for evaluating the robustness of verifier trained using the SRC reward (Section 4.3.1).
### 5.2 ITERATIVE GOODHARTING TRAINING
We run an alternative training scheme where the prover does not have an assigned role. It is trained with reward (4), which is just the verifier score. We call this the ‘goodharting’ baseline because usually during Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022), the prover is directly optimized to maximize the score of a reward model. When the reward model is not robust enough, it is easy to over-optimize so the prover overfits to the imperfect proxy reward. This
![Figure 10 shows five plots illustrating the iterative goodharting prover training dynamics across verifier sizes 10^-4 to 10^0. The plots show accuracy versus RL steps for round 0 and round 5. The verifier sizes are 10^-4, 10^-3, 3x10^-3, 10^-2, and 1.0. The plots show that larger verifiers generally achieve better maximum accuracy and reach robustness faster.](71ab4df17511d75261da8d462d643b1a_img.jpg)
| Verifier Size (Goodhart Reward) | Round   | Peak Accuracy | Approx. RL Step at Overoptimization |
|---------------------------------|---------|---------------|-------------------------------------|
| $10^{-4}$                       | Round 0 | ~0.7          | ~1.5k                               |
| $10^{-4}$                       | Round 5 | ~0.7          | ~3k                                 |
| $10^{-3}$                       | Round 0 | ~0.7          | ~1.5k                               |
| $10^{-3}$                       | Round 5 | ~0.7          | ~4k                                 |
| $3 \times 10^{-3}$              | Round 0 | ~0.7          | ~2k                                 |
| $3 \times 10^{-3}$              | Round 5 | ~0.7          | > 8k                                |
| $10^{-2}$                       | Round 0 | ~0.7          | ~4k                                 |
| $10^{-2}$                       | Round 5 | ~0.8          | > 8k                                |
| $1.0$                           | Round 0 | ~0.8          | ~4k                                 |
| $1.0$                           | Round 5 | ~0.8          | > 8k                                |
Figure 10 shows five plots illustrating the iterative goodharting prover training dynamics across verifier sizes 10^-4 to 10^0. The plots show accuracy versus RL steps for round 0 and round 5. The verifier sizes are 10^-4, 10^-3, 3x10^-3, 10^-2, and 1.0. The plots show that larger verifiers generally achieve better maximum accuracy and reach robustness faster.
Figure 10: **Iterative goodharting prover training dynamics across verifier sizes  $10^{-4}$  to  $10^{0}$ .** Overoptimization happens later at higher rounds, with some degradation for one verifier size ( $10^{-3}$ ). Larger verifiers lead to better maximum prover accuracy and reach better robustness in fewer rounds.
phenomenon is called ‘goodharting’ or ‘reward hacking’ (Gao et al., 2023; Skalse et al., 2022). The verifier can learn from the incorrect and correct samples from the same prover.
In more details, for each round, we train the prover using PPO with reward (4) until its training accuracy drops below 0.2. Next, we sample balanced correct and incorrect samples on the verifier split of the training data for all prover checkpoints throughout PPO. The samples are then mixed into the verifier training dataset for the next round. We still keep the total number of samples per problem constant for training the verifier in each round.
Figure 10 shows the training dynamics. We early stop each round when the prover’s training accuracy tanks below 0.2 on average in the last 80 RL steps. Across verifier sizes  $10^{-4}$  to  $10^{0}$  of the prover, the number of steps it takes for the prover to over-optimize generally increase. Up to size  $10^{-2}$ , the larger the verifier is, the fewer number of training rounds it requires to reach perfect goodharting robustness within 8k RL steps. Verifiers of medium size ( $3\times 10^{-3}$  ~  $10^{-2}$ ) benefit most from the iterative goodharting training, where as smaller ( $\le 10^{-3}$ ) and larger ( $10^{0}$ ) verifiers benefit less. We further evaluated the robustness of verifier trained using goodharting reward under attacks from the SRC reward (2) and the alternative CGC reward (3), and the verifiers’ robustness go up sometimes but not always (Figures S7a, S7b). The balanced best-of- $n$  robustness increases (Figure S8).
However, when we showed the samples from the goodharting training to human trainers, there is no significant difference in human judge accuracy across rounds (Fig. S6), although Fig. S6 is from a pilot study that has some design flaws (see Appendix I). Overall, our results on the iterative goodharting training show that it increases verifier soundness to a certain extent, but we do not have enough evidence showing that it improves human legibility.
