# PROVER-VERIFIER GAMES IMPROVE LEGIBILITY OF LLM OUTPUTS
Jan Hendrik Kirchner\* Yining Chen\*
Harri Edwards† Jan Leike† Nat McAleese Yuri Burda†
OpenAI

## C WHEN THE VERIFIER IS TOO SMALL OR TOO LARGE, CHECKABILITY TRAINING IS INHIBITED.
### a very small verifier (verifier : policy = $10^{-4}$ )
![Icon of a person's head.](0ccbf2e1f1d9d0aae8865d824a1fc322_img.jpg)
Icon of a person's head.
Shawna's father is five times as old as Shawna. Shawna is currently three times as old as Aliya. If Aliya is 3 years old, how old is Shawna's father?
![Icon of a robot head.](daff947a443eb2a4735e3a81b6f756d9_img.jpg)
Icon of a robot head.
First, let us define some useful concepts to help better understand about some important concepts. Let us explain about some useful concepts. Let us introduce some useful notations to explain about some useful labels. Let us help better comprehend about some concepts. Let us firstly define some useful notations to help better understand some useful naming conventions. Let us firstly introduce some useful naming conventions to explain about some useful notations to help better understand some useful concepts. [...]
Similarly, Shawna is currently three times as old as Aliya. Hence, this means, similarly, thus, this means, the number of dollars for Shawna is  $N_Q = 3 \times 5 = \text{<<}3 \times 5=190\text{>>}190$  dollars.
Now, to compute for the sum of money. So, this means, this means, this means, this means, the sum of dollars is  $N_P + N_Q$ , or, 300 (from above) + 190 (from above), resulting in  $300 + 190 = \text{<<}300+190 = 385\text{>>}385$  dollars.
And lastly, Aliya is 3 dollars. Hence, to compute for the total dollars. Finally, finally, to compute the total dollars, we finally, we compute, to compute, finally, this, 385 dollars.
Therefore he paid a total of 385 dollars.
Answer: 385
### b
![](6e63447db1c11224a74c7add39be1b4e_img.jpg)
**helpful**
| Fraction of round completed | initialization | round 0 | round 1 | round 2 | round 3 | round 4 |
|-----------------------------|----------------|---------|---------|---------|---------|---------|
| 0                           | 0.45           | 0.60    | 0.60    | 0.60    | 0.60    | 0.60    |
| 0.2                         | 0.45           | 0.65    | 0.62    | 0.61    | 0.60    | 0.58    |
| 0.4                         | 0.45           | 0.60    | 0.58    | 0.55    | 0.52    | 0.50    |
| 0.6                         | 0.45           | 0.55    | 0.50    | 0.45    | 0.42    | 0.40    |
| 0.8                         | 0.45           | 0.50    | 0.45    | 0.40    | 0.48    | 0.35    |
| 1.0                         | 0.45           | 0.45    | 0.40    | 0.35    | 0.60    | 0.30    |
  
**sneaky**
| Fraction of round completed | initialization | round 0 | round 1 | round 2 | round 3 | round 4 |
|-----------------------------|----------------|---------|---------|---------|---------|---------|
| 0                           | 0.45           | 0.50    | 0.50    | 0.50    | 0.50    | 0.50    |
| 0.2                         | 0.45           | 0.60    | 0.55    | 0.52    | 0.50    | 0.48    |
| 0.4                         | 0.45           | 0.50    | 0.40    | 0.30    | 0.20    | 0.10    |
| 0.6                         | 0.45           | 0.40    | 0.10    | 0.05    | 0.02    | 0.01    |
| 0.8                         | 0.45           | 0.10    | 0.05    | 0.02    | 0.01    | 0.01    |
| 1.0                         | 0.45           | 0.05    | 0.02    | 0.01    | 0.01    | 0.01    |
Supplementary Figure S4: **A very small verifier makes training dynamics unstable and results in degenerate solutions.** **a** Sample completion from round zero of legibility training with a very small verifier. **b** Average accuracy for the helpful (top) and sneaky (bottom) provers as a function of the completed fraction of each round.
![](bfb6d182d624680db577069bbc0b2a93_img.jpg)
**a** **large verifier** (verifier : policy = 1)
Shawna's father is five times as old as Shawna. Shawna is currently three times as old as Aliya. If Aliya is 3 years old, how old is Shawna's father?
First, you begin by figuring out the age of Shawna. Since it is stated in the problem that Shawna is currently "three times as old as Aliya" and since the information about Aliya is presented, that means Shawna is currently  $3 * 3 = \lt \lt 3 * 3 = 9 \gt \gt 9$  years old
Next, you are able to find the age of Shawna's father. Since it is stated in the problem that Shawna's father is "five times as old as Shawna", and since you know the current age of Shawna, you can determine that the age of Shawna's father is  $5 * 9 = \lt \lt 5 * 9 = 45 \gt \gt 45$
Therefore, the final solution is 45.
Answer: 45
**b**
Legend: initialization (dotted line), round 0, 1, 2, 3, 4 (solid lines)
**Helpful Prover (Top Plot)**
| Fraction of round completed | 0    | 0.2  | 0.4  | 0.6  | 0.8  | 1.0  |
|-----------------------------|------|------|------|------|------|------|
| Accuracy (Rounds 0-4)       | ~0.4 | ~0.6 | ~0.7 | ~0.8 | ~0.8 | ~0.8 |
**Sneaky Prover (Bottom Plot)**
| Fraction of round completed | 0    | 0.2  | 0.4  | 0.6  | 0.8  | 1.0  |
|-----------------------------|------|------|------|------|------|------|
| Round 0 Accuracy            | ~0.2 | ~0.1 | ~0.4 | ~0.5 | ~0.5 | ~0.4 |
| Rounds 1-4 Accuracy         | ~0.2 | ~0.1 | ~0.1 | ~0.2 | ~0.2 | ~0.2 |
**c**
| Round                    | init. | 0    | 1    | 2    | 3    |
|--------------------------|-------|------|------|------|------|
| Fraction correct (human) | 0.86  | 0.72 | 0.78 | 0.75 | 0.77 |
Note: All bars in Figure c include error bars. The 'init.' bar is grey, while rounds 0-3 are red.
Supplementary Figure S5: **A verifier as large as the prover does not increase in robustness over rounds and results in degenerate solutions.** **a** Sample completion from round zero of legibility training with a very small verifier. **b** Average accuracy for the helpful (top) and sneaky (bottom) provers as a function of the completed fraction of each round. **c** Median accuracy of time-constrained human contractors when evaluating solutions from the helpful prover at different rounds of checkability training. This plot is from a potentially flawed campaign (see Appendix I).