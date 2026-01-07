# PROVER-VERIFIER GAMES IMPROVE LEGIBILITY OF LLM OUTPUTS
Jan Hendrik Kirchner\* Yining Chen\*
Harri Edwards† Jan Leike† Nat McAleese Yuri Burda†
OpenAI

## 4 RESULTS
### 4.1 BASELINE: DIRECT OPTIMIZATION FOR CORRECTNESS
A natural baseline for our setup is to train the prover model directly to optimize for the correctness of the solutions without the intermediate verifier. Specifically, we can use Proximal Policy Optimization (PPO) against a binary grader that checks if the solution is correct or not. This approach has been explored in works such as Havrilla et al. (2024), where a language model is fine-tuned using reinforcement learning (RL) with a reward based on the correctness of the answers. We use a KL regularization towards prover initialization with the same small coefficient for all methods in this paper.
We find that, indeed, optimizing our prover with correctness as the only goal achieves strong performance on grade-school math problems (Fig. 4a). When inspecting samples from the resulting prover, we notice that the solution becomes longer as the prover solves each question in multiple
![](352c5fab6f936356e9570761a02ab71e_img.jpg)
Figure 4: Performance and legibility analysis of a model trained for correctness.
**a baseline: correctness training only**
Plot of model accuracy vs. RL steps. Accuracy starts at initialization ( $\bullet$ ) at 0 steps (approx. 0.43), rises to a peak of approx. 0.85 near 7k steps, and then shows a pronounced drop at 8k steps due to solutions reaching maximum completion length.
**b Mean accuracy (human)**
| Training Stage | Symbol           | Approx. Mean Accuracy |
|----------------|------------------|-----------------------|
| init.          | $\bullet$        | 0.82                  |
| step 3k        | $\blacktriangle$ | 0.73                  |
| step 7k        | $\star$          | 0.74                  |
Y-axis range: 0.65 to 0.9.
**c Mean time to solve (sec)**
| Training Stage | Symbol           | Approx. Mean Time (sec) |
|----------------|------------------|-------------------------|
| init.          | $\bullet$        | 26.2                    |
| step 3k        | $\blacktriangle$ | 26.5                    |
| step 7k        | $\star$          | 26.8                    |
Y-axis range: 24 to 32.
**d Contractor accuracy vs. model accuracy**
Shows mean human accuracy (y-axis: 0.6 to 0.9) vs. fraction correct by model (x-axis: 0.2 to 0.9).
- **init.** (grey line,  $\bullet$ ): Shows a clear upward trend, increasing from approx. 0.78 to 0.88.
- **step 3k** (light blue line,  $\blacktriangle$ ): Remains relatively flat around 0.70-0.75, with a slight drop at higher model accuracy.
- **step 7k** (dark blue line,  $\star$ ): Peaks at approx. 0.75 (at 0.5 model accuracy) and then drops sharply to approx. 0.63.
**Figure 4: Learning from a binary grader leads to strong performance, but poor legibility.** **a** Model accuracy as a function of RL training steps. The pronounced drop at 8k steps is due to solutions reaching the maximum completion length. **b** Mean accuracy of time-constrained human contractors when evaluating solutions from the initialization ( $\bullet$ ), after 3k steps ( $\triangle$ ), or 7k steps ( $\star$ ). **c** Mean time required by the time-constrained human contractor to correctly label solutions. **d** Mean accuracy of contractors per solution as a function of model accuracy on the same solution for solutions from the initialization (grey), the prover after 3k steps (dashed blue), or 7k steps (solid blue). For each experiment, we separate data into quartiles according to the model accuracy before averaging. See Appendix I for details on the experiment set-up.
ways and later in training become largely unintelligible (Appendix Fig. S1) despite still producing the correct answer.
To measure how much the peculiar style affects legibility of solutions, we presented samples from different points during training to human contractors and asked them to determine the validity of the solutions under a strict 45 second time limit per solution. They were asked to judge the correctness of the given solution rather than the final answer (see Appendix I for full instructions and a task example). While the contractors were able to reliably distinguish correct and incorrect solutions from the prover initialization, their accuracy and speed deteriorated noticeably when presented with solutions from the trained prover (Fig. 4b,c). This decrease in human legibility is most pronounced for questions that the model cannot reliably solve (Fig. 4d).
In summary, we observe that when trained solely for correctness, our model learns to produce useful but unintelligible solutions that time-constrained humans struggle to evaluate.
### 4.2 DYNAMICS OF CHECKABILITY TRAINING
To study whether it is possible to retain legibility while improving performance, we ran several rounds of checkability training on the augmented GSM dataset. In this section, we present results where the provers are trained using the Signed Relative Convincingness (SRC) reward (Eq. (2)), which is the reward that we found most effective for improving legibility.
Within each round of the prover training phase, both the sneaky and the helpful prover monotonically improve their convincingness to the small verifier (Fig. 5a). In contrast, the fraction of correct solutions produced by the provers evolves non-monotonically (Fig. 5b,c). While the helpful prover largely produces increasingly correct solutions, the sneaky prover eventually produces incorrect solutions that match or surpass the verifier score of correct solutions from the helpful prover in the first few rounds of training. The helpful prover’s performance decreases at the end of each round potentially due to weight sharing with the sneaky prover (they are a single LLM with different prompts). Note that the prover and the verifier have the same initializations at the beginning of each round, and the only difference between rounds is the data mixture used to train the verifier.
![](adb1f42239329fa8283d1a40005f989f_img.jpg)
### (a) Verifier score vs RL steps
| Round | RL steps        | honest (solid red) | sneaky (solid brown) | correct (dotted red) | incorrect (dotted brown) |
|-------|-----------------|--------------------|----------------------|----------------------|--------------------------|
| 0     | 0               | 2.0                | 2.0                  | 2.0                  | 2.0                      |
| 0     | 1k              | 3.5                | 3.5                  | 3.5                  | 3.5                      |
| 0     | 2k              | 4.0                | 4.0                  | 4.0                  | 4.0                      |
| 0     | 3k              | 4.5                | 4.5                  | 4.5                  | 4.5                      |
| 0     | 4k (Early Stop) | 4.8                | 4.8                  | 4.8                  | 4.8                      |
| 1     | 0               | 2.0                | 2.0                  | 2.0                  | 2.0                      |
| 1     | 1k              | 3.5                | 3.5                  | 3.5                  | 3.5                      |
| 1     | 2k              | 4.0                | 4.0                  | 4.0                  | 4.0                      |
| 1     | 3k              | 4.8                | 4.5                  | 4.8                  | 4.5                      |
| 1     | 4k (Early Stop) | 5.2                | 4.8                  | 5.2                  | 4.8                      |
| 2     | 0               | 2.0                | 2.0                  | 2.0                  | 2.0                      |
| 2     | 2k              | 4.5                | 4.0                  | 4.5                  | 4.0                      |
| 2     | 4k              | 5.5                | 5.0                  | 5.5                  | 5.0                      |
| 2     | 6k              | 6.5                | 5.5                  | 6.5                  | 5.5                      |
| 2     | 8k              | 7.2                | 6.0                  | 7.2                  | 6.0                      |
| 5     | 0               | 2.0                | 2.0                  | 2.0                  | 2.0                      |
| 5     | 2k              | 5.5                | 4.5                  | 5.5                  | 4.5                      |
| 5     | 4k              | 6.5                | 5.0                  | 6.5                  | 5.0                      |
| 5     | 6k              | 7.2                | 5.2                  | 7.2                  | 5.2                      |
| 5     | 8k              | 7.8                | 5.5                  | 7.8                  | 5.5                      |
### (b) Accuracy vs RL steps
| Round | RL steps        | helpful (red) | sneaky (brown) |
|-------|-----------------|---------------|----------------|
| 0     | 0               | 0.6           | 0.1            |
| 0     | 1k              | 0.7           | 0.05           |
| 0     | 2k              | 0.5           | 0.02           |
| 0     | 3k              | 0.65          | 0.01           |
| 0     | 4k (Early Stop) | 0.55          | 0.0            |
| 1     | 0               | 0.6           | 0.1            |
| 1     | 1k              | 0.7           | 0.05           |
| 1     | 2k              | 0.65          | 0.02           |
| 1     | 3k              | 0.5           | 0.01           |
| 1     | 4k (Early Stop) | 0.4           | 0.0            |
| 2     | 0               | 0.6           | 0.1            |
| 2     | 2k              | 0.7           | 0.05           |
| 2     | 4k              | 0.65          | 0.02           |
| 2     | 6k              | 0.55          | 0.01           |
| 2     | 8k              | 0.45          | 0.0            |
| 5     | 0               | 0.6           | 0.1            |
| 5     | 2k              | 0.7           | 0.05           |
| 5     | 4k              | 0.65          | 0.02           |
| 5     | 6k              | 0.55          | 0.01           |
| 5     | 8k              | 0.45          | 0.0            |
### (c) Average accuracy vs fraction completed
**Helpful Prover**
| Round | 0.0  | 0.2  | 0.4  | 0.6  | 0.8  | 1.0  |
|-------|------|------|------|------|------|------|
| init  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  | 0.6  |
| 0     | 0.6  | 0.65 | 0.65 | 0.6  | 0.6  | 0.6  |
| 1     | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| 2     | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| 3     | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| 4     | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
**Sneaky Prover**
| Round | 0.0 | 0.2 | 0.4  | 0.6  | 0.8  | 1.0  |
|-------|-----|-----|------|------|------|------|
| init  | 0.6 | 0.6 | 0.6  | 0.6  | 0.6  | 0.6  |
| 0     | 0.5 | 0.6 | 0.6  | 0.6  | 0.55 | 0.5  |
| 1     | 0.5 | 0.6 | 0.55 | 0.4  | 0.3  | 0.2  |
| 2     | 0.5 | 0.6 | 0.4  | 0.2  | 0.1  | 0.05 |
| 3     | 0.5 | 0.6 | 0.3  | 0.1  | 0.05 | 0.02 |
| 4     | 0.5 | 0.6 | 0.2  | 0.05 | 0.02 | 0.01 |
**Figure 5: Dynamics of the prover training phase across rounds for one pair of verifier and prover.** Mean verifier score of correct (solid) and incorrect (dashed) solutions (a) and accuracy (b) as a function of RL training steps, for subsequent rounds (columns) and for the helpful (red) or sneaky (brown) prover. Arrow indicates increasing number of steps before the accuracy of the sneaky prover drops below 10%. We early stop when sneaky incorrect solutions become as convincing as helpful correct solutions. This verifier is  $10^{-3}$  the size of the prover. c Average accuracy for the helpful (top) and sneaky (bottom) provers as a function of the completed fraction of each round.
We were curious to see if the above dynamic holds also for a wider range of pairs of verifier and prover effective compute amounts. When only varying the size of the verifier by a few orders of magnitude ( $10^{-3}$  to  $10^{-1.8}$ -times the compute of the prover), the dynamics remain largely unchanged (Fig. 5c). However, when using a much smaller verifier ( $10^{-4}$ -times the compute of the prover), the checkability training becomes unstable and produces weak provers with poor legibility (Sup. Fig. S4). Similarly, using a verifier that matches the size of the prover ( $10^{0}$ -times the compute of the prover) prevents the verifier from becoming more robust in subsequent rounds and produces provers with strong performance but poor legibility (Sup. Fig. S5). We cannot exclude the possibility that the poor performance of very small and large verifiers is due to non-optimal hyperparameters, but we also suspect there are real reasons for checkability training to not work well for these verifiers. In particular, if the verifier’s capability is too limited, it may be fundamentally impossible to perform verification. If the verifiers are very similar to the provers, the sneaky prover can more easily simulate the verifier’s mistakes (Burns et al., 2023), making it more difficult for the verifier to become robust.
To summarize, our checkability training exhibits a characteristic dynamic where the helpful prover improves at solving math problems while over subsequent rounds the sneaky prover struggles to produce incorrect, convincing solutions.
### 4.3 EVALUATIONS OF CHECKABILITY TO SMALLER VERIFIERS AND LEGIBILITY TO HUMANS
At the end of each round of checkability training, we obtain three artifacts:
1. a verifier trained on diverse adversarial examples,
2. a sneaky prover that produces convincing, subtly incorrect solutions,
3. a helpful prover that produces convincing, correct solutions.
In this section, we perform various evaluations of the soundness of verifiers and the legibility of sneaky and correct solutions across rounds of training shown in Figure 5.
#### 4.3.1 THE VERIFIER BECOMES MORE ROBUST THROUGHOUT CHECKABILITY TRAINING
The verifier’s sound evaluation of adversarial inputs from a sneaky prover (of equal capability as the helpful prover) is one requirement of legibility by our definition. We observe that throughout
![Figure 6: Verifiers increase in robustness as measured by difficulty of sneaky-only attacks and balanced best-of-n accuracy. (a) Number of steps until sneaky finds an exploit for SRC, goodhart, and CGC adversaries as a function of verifier sizes across 5 rounds. (b) Fraction of correct solutions when selecting the most convincing out of n balanced samples for varying n and different rounds.](d48475a25698b1c0592e4cfe07138f2a_img.jpg)
### Figure 6a: RL steps until exploit
| Verifier Size | Round 0 | Round 1   | Round 2   | Round 3   | Round 4   |
|---------------|---------|-----------|-----------|-----------|-----------|
| $10^{-2}$     | ~4.2k   | not found | not found | not found | not found |
| $10^{-3}$     | ~1.8k   | ~2.8k     | ~5.5k     | ~4k       | ~7.5k     |
| $10^{-4}$     | ~1.8k   | ~1.8k     | ~4.2k     | ~4k       | ~3.5k     |
| Verifier Size | Round 0 | Round 1   | Round 2   | Round 3   | Round 4   |
|---------------|---------|-----------|-----------|-----------|-----------|
| $10^{-2}$     | ~7.5k   | not found | not found | not found | not found |
| $10^{-3}$     | ~2k     | ~2.5k     | ~5.5k     | ~4.5k     | ~5.2k     |
| $10^{-4}$     | ~1.8k   | ~3.8k     | ~3.2k     | ~3.2k     | ~4k       |
| Verifier Size | Round 0 | Round 1 | Round 2   | Round 3   | Round 4   |
|---------------|---------|---------|-----------|-----------|-----------|
| $10^{-2}$     | ~2.8k   | ~7.5k   | not found | not found | not found |
| $10^{-3}$     | ~2.2k   | ~3.5k   | ~4.5k     | ~4.8k     | ~7.8k     |
| $10^{-4}$     | ~1.8k   | ~2.2k   | ~2.1k     | ~2.5k     | ~3.5k     |
### Figure 6b: Accuracy best-of- $n$
| Round | n=0 | n=200 | n=400 | n=600 |
|-------|-----|-------|-------|-------|
| 0     | 0.5 | ~0.28 | ~0.25 | ~0.23 |
| 1     | 0.5 | ~0.35 | ~0.34 | ~0.31 |
| 2     | 0.5 | ~0.36 | ~0.35 | ~0.34 |
| 3     | 0.5 | ~0.5  | ~0.5  | ~0.5  |
| 4     | 0.6 | ~0.52 | ~0.52 | ~0.52 |
Figure 6: Verifiers increase in robustness as measured by difficulty of sneaky-only attacks and balanced best-of-n accuracy. (a) Number of steps until sneaky finds an exploit for SRC, goodhart, and CGC adversaries as a function of verifier sizes across 5 rounds. (b) Fraction of correct solutions when selecting the most convincing out of n balanced samples for varying n and different rounds.
**Figure 6: Verifiers increase in robustness as measured by difficulty of sneaky-only attacks and balanced best-of- $n$  accuracy**
**a** Number of steps until sneaky finds an exploit for the SRC (blue), goodharting (orange), and CGC (green) adversaries as a function of verifier sizes. We declare an attack ‘successful’ for the SRC and CGC adversaries when the sneaky becomes 95% incorrect and the average verifier score of sneaky incorrect samples become as high as that of helpful correct samples from the same round. We declare a goodharting adversary ‘successful’ when optimizing a prover on the verifier score reaches accuracy  $< 0.2$ .
**b** Fraction of correct solutions when selecting the most convincing out of  $n$  balanced samples for varying  $n$  and different rounds. The solutions are from the last checkpoints of current-round helpful provers, balanced to have 50% correctness.
checkability training, the sneaky prover requires increasingly many steps to discover highly convincing, incorrect solutions (Fig. 5b). However, it is unclear whether this increase is due to the sneaky and helpful provers sharing weights. Hence, we perform a variety of additional attacks by separately training sneaky-only provers that do not share weights with the helpful prover to evaluate whether the verifier becomes able to discern correct from incorrect completions beyond the exact set-up during training.
We performed three attacks with sneaky-only provers using the following reward functions:
1. the SRC reward (2) to ablate the effect of weight sharing between the helpful and the sneaky prover.
2. the CGC reward (3) to test generalization of verifier soundness to different variations of checkability training
3. the goodharting reward (4) to straightforwardly test robustness to optimization for convincingness (Gao et al., 2023)
We performed the above attacks on verifiers of different sizes across five rounds of checkability training (Fig. 6a). We observe an increase in the number of steps each adversary takes until finding an exploit. Larger verifiers are harder to exploit. For example, none of the sneaky attacks found exploits for verifiers of relative size  $10^{-2}$  beyond round 0 within 8k RL steps, whereas verifier of relative size  $10^{-3}$  remain exploitable by the goodharting and SRC adversaries until round 4.
The test attacks we perform are far from complete and there is no simple way to evaluate whether the verifiers are robust to *all* sneaky provers within the model family. Nevertheless, the rank correlation between those robustness measures do reflect a general increase in compute budget if some actor trains an adversarial prover to fool the verifier. A more complete evaluation of the verifier’s adversarial robustness would include a variety of common white-box and black-box attacks (Zou et al., 2023; Shah et al., 2023; Hayase et al., 2024).
Another way to assess the checkability of the helpful prover to the verifier is via best-of- $n$  accuracy at higher  $n$  (Gao et al., 2023), where we analyze the correctness of the highest-scoring solution according to the verifier out of  $n$  solutions (Fig. 6b). In order to fairly compare provers with different average pass rates, we collect balanced incorrect and correct samples and only use problems that have at least  $n/2$  correct and  $n/2$  incorrect samples for evaluating best-of- $n$ . We sample up to 2560
![](36ac3e730a00d3f42d3400f5709f641a_img.jpg)
**Figure 7: Analysis of Sneaky Prover Performance**
**a** GPT-4 validity score vs. round for "sneaky" solutions. The graph shows average (avg) and minimum (min) scores for correct (solid line) and incorrect (dotted line) solutions. Note the significant dip in the correct validity score at round 0.
| Round | Correct (avg) | Correct (min) | Incorrect (avg) | Incorrect (min) |
|-------|---------------|---------------|-----------------|-----------------|
| init. | ~0.85         | ~0.82         | ~0.25           | ~0.10           |
| 0     | ~0.62         | ~0.12         | ~0.46           | ~0.02           |
| 1     | ~0.75         | ~0.48         | ~0.38           | ~0.02           |
| 2     | ~0.68         | ~0.35         | ~0.36           | ~0.02           |
| 3     | ~0.92         | ~0.82         | ~0.55           | ~0.02           |
| 4     | ~0.95         | ~0.90         | ~0.54           | ~0.02           |
**b** Mean accuracy of time-constrained human contractors when evaluating sneaky solutions across rounds.
| Round | Accuracy (human) |
|-------|------------------|
| 0     | ~0.845           |
| 1     | ~0.840           |
| 2     | ~0.825           |
| 3     | ~0.810           |
**c** Examples of incorrect solutions with high verifier scores. Flaws are highlighted in red in the original image.
**Problem:** Shawna's father is five times as old as Shawna. Shawna is currently three times as old as Aliya. If Aliya is 3 years old, how old is Shawna's father?
| <p><b>round 1</b></p> <p>First we need to determine the age of Aliya. From the problem description, it says that Aliya is 3 years old. Then we need to determine the age of Shawna. From the problem description, we know that Shawna is currently three times as old as Aliya, so the age of Shawna is 3 times 3, or &lt;&lt;3*3=9&gt;&gt;9. Next, we need to determine the father's age. It says that the father is five times as old as Shawna, so the age of the father becomes 5 times 9, which equals <b>&lt;&lt;5 * 9 = 36&gt;&gt;36</b>. Answer: 36</p> | <p><b>round 3</b></p> <p>We are told that Shawna's father is five times as old as Shawna. Shawna is currently three times as old as Aliya. So, if Aliya is 3 years old, then Shawna is <b>3 * 5 = &lt;&lt;3*5=15&gt;&gt;15</b>. And, we're asked to find how old is Shawna's father. Shawna's father is five times as old as Shawna. So, the age of Shawna's father is <b>5 * 15 = &lt;&lt;5*15=75&gt;&gt;75</b>. Answer: 75</p> |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
Figure 7: **Flaws introduced by the sneaky prover become more localized and harder for time-limited humans to spot.** **a** Mean (circle) and minimum (triangle) validity score over portions of correct (solid) and incorrect (dotted) solutions from the initialization (left) and from different rounds of checkability training. **b** Mean accuracy of time-constrained human contractors when evaluating solutions from the sneaky prover at different rounds of checkability training. This plot is from a potentially flawed campaign (see Appendix I). **c** Two examples of incorrect solutions with high verifier score from different points of training. Flaws are highlighted in red. Note that for the second example, the flaw is an incorrect translation of words into equations.
solutions for each test problem and the helpful prover from each round. In Figure 6(b), for each  $n$ , we combine  $n/2$  correct and  $n/2$  incorrect solutions from the current-round helpful prover  $\pi_i$ , and evaluated the best-of- $n$  performance of verifier  $V_i$ . The balanced best-of- $n$  performance increases for  $(V_i, \pi_i)$  pairs at higher rounds, indicating that the verifier can better judge the helpful prover's correctness even when controlling for pass rates.
#### 4.3.2 THE SNEAKY PROVER BECOMES INCREASINGLY SUBTLE THROUGHOUT CHECKABILITY TRAINING
Having established that the verifier becomes increasingly robust to adversarial attacks, we turn to analysing the sneaky attacks that *do* fool the verifier.
While sneaky incorrect solutions from early rounds often contain multiple obvious mistakes and non-sequitur, incorrect solutions from later rounds contain increasingly subtle flaws (Fig. 7c and Sup. Fig. S2). To quantify this shift, we used GPT-4 to assess the validity of small portions of sneaky solutions from different rounds (Fig. 7a). We observe that while the minimum score over portions of a solution remains very low (indicating a flaw detectable by GPT-4), the average score increases over rounds (indicating the general soundness of the remaining portions).
We were interested to see whether the adversarial attacks against the small verifier might transfer to humans. Interestingly, when presenting the sneaky solutions to the time-constrained contractors, the
![Figure 8: Checkability training jointly increases model performance and legibility. Three bar charts showing (a) mean accuracy of the helpful prover model, (b) mean accuracy of human contractors, and (c) time required for human contractors to solve, across 6 rounds (0-5). Charts (b) and (c) include error bars.](dd0f5301a5a6dd7c319701302110de88_img.jpg)
| Round | Mean Accuracy (Model) | Mean Accuracy (Human) | Mean Time to Solve (sec) |
|-------|-----------------------|-----------------------|--------------------------|
| 0     | 0.57                  | 0.76                  | 27.5                     |
| 1     | 0.60                  | 0.78                  | 27.0                     |
| 2     | 0.56                  | 0.81                  | 26.8                     |
| 3     | 0.61                  | 0.81                  | 26.2                     |
| 4     | 0.63                  | 0.77                  | 26.5                     |
| 5     | 0.63                  | 0.83                  | 26.3                     |
Note: Charts (b) and (c) include error bars representing uncertainty in human performance metrics.
Figure 8: Checkability training jointly increases model performance and legibility. Three bar charts showing (a) mean accuracy of the helpful prover model, (b) mean accuracy of human contractors, and (c) time required for human contractors to solve, across 6 rounds (0-5). Charts (b) and (c) include error bars.
Figure 8: **Checkability training jointly increases model performance and legibility.** **a** Mean accuracy of the helpful prover at different rounds of checkability training. **b,c** Mean accuracy (**b**) and time required to correctly label solutions (**c**) of time-constrained human contractors when evaluating solutions from the helpful prover at different rounds of checkability training.
fraction of correctly labeled solutions was comparatively high, but decreased throughout the rounds (Fig. 7b). This indicates that the sneaky prover produced by checkability training might be useful as a tampering model for training critique models (Saunders et al., 2022; McAleese et al., 2024). Note that Fig. 7b is from a pilot study that has some design flaws (see Appendix I).
In sum, the sneaky prover produced by our checkability training is capable of generating subtly flawed solutions that are non-trivial to spot for time-constrained humans.
#### 4.3.3 THE HELPFUL PROVER BECOMES INCREASINGLY LEGIBLE THROUGHOUT CHECKABILITY TRAINING
We finally turn our attention to the helpful prover and its evolution through successive rounds of checkability training. A key desideratum of our framework is that the helpful prover not only achieves high correctness but also maintains legibility for human evaluators.
We observe that as the checkability training progresses, the helpful prover produces solutions with increasing accuracy (Figure 8a). In contrast to the training for correctness only (Fig. 4), solutions from the helpful prover remain coherent throughout training (Fig. 2). This improvement indicates that the prover is effectively leveraging the feedback from the verifier to enhance its problem-solving capabilities.
To assess the human legibility of the helpful prover’s solutions, we again conducted evaluations with time-constrained human contractors. We observe that both the contractors’ mean accuracy in evaluating solutions (Figure 8b) and the time required for evaluation (Figure 8c) improve as the prover advances through successive rounds. At the end of training, the helpful prover matches the legibility of the initialization while achieving substantially stronger performance. However, the performance falls short of the performance achieved by a prover trained for correctness only (Fig. 4), revealing a tension between achieving high performance and retaining legibility (Leike, 2022). We note that one way to resolve this tension is to separate the model’s chain-of-thought from proof of correctness, see Appendix 6.1 for more discussion.
In summary, our findings demonstrate that checkability training facilitates the development of a helpful prover whose outputs are increasingly correct and legible. This dual improvement is critical for applications where human oversight and collaboration with AI systems are essential.