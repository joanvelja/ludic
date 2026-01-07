# Prover-verifier game-inspired paper
We want to implement a training loop that is inspired by the prover-verifier game. The prover-verifier game is a three-player game where two players (the provers) take as an input a query Q, and are pitted adversarially against each other: the honest prover is tasked to provide a correct answer to the query, while the dishonest prover is tasked to provide an intentionally wrong answer to the query. The third player (the verifier) is tasked to recognize whether the answer is correct or not. The honest prover wins if the verifier scores its solution higher than the dishonest prover, and the dishonest prover wins vice versa.

The paper can be found at @research/papers/arXiv-pvg/. The directory contains all the sections of the paper as separate folders, each containing the markdown file for that section, as well as any additional files that are needed for that section.

**IMPORTANT NOTE**: We do not really care about legibility per-se in my project, but rather about the adversarial aspect of the game. Your utmost focus should be towards understanding the intricacies of the game, and how to best implement it within the repository.

I think this is a very compelling paper that leverages adversarial games to train legible models "by default", namely as a byproduct of this training regime. One thing that is underlooked here is that the verifier does also get much better over the course of training, at least in principle. 
Something that I think the paper does not address though, or that I think can become a huge problem, is the concept of what I'll call *spurious equilibria*. Let me explain this in my words. 

The way the models are trained is quite straightforward: honest and sneaky provers obtain as an input the query, and each one has to provide a solution to that problem, conditioned on the role of the model (be it correct —honest— or incorrect —sneaky). The solutions are then passed onto the verifier, which is trained via a Bradley-Terry (RLHF-like) objective, to prefer the honest solutions from the sneaky ones. What this creates, is a model that can produce scalar scores for any given input, what the paper calls "convincingness scores", i.e., how "good" does a solution look to the verifier. In my view though, a big problem here is the fact that if any spurious feature correlates heavily with the honest solutions, but not with the sneaky ones, the verifier will learn to score on the basis of that spurious feature (easier to learn) rather than on the basis of the notion of correctness. I thus maintain that, empirically, the space of tokens is too large for the sneaky prover to navigate such that the theoretical equilibria stated in the paper holds. In a way, once the GAN-like learning loop collapses, the sneaky prover has little traction for improving its reward! Appendix C and Figure S3 provide direct evidence: they trained a tiny discriminator to classify helpful vs. sneaky samples, achieving near-perfect ROC AUC (~1.0) in early rounds. This means the verifier can "solve" its task by learning superficial stylistic signatures rather than actual correctness reasoning.

The paper's key observation is that this separation *decreases* over rounds with the SRC reward (but persists with CGC). However, this self-correction is empirically observed, not theoretically guaranteed—and it's unclear whether it would hold at scale or in harder domains.

So a big priority in my research project is to figure out how to provably avoid (or mitigate) these spurious equilibria, and an approach I had thought about is the idea of playing with information asymmetry. We can leverage sequentiality (as opposed to simultaneity) in the provers' moves to account for this: the honest prover moves first, producing a solution z_h to a problem q. Then the sneaky prover plays and produces z_s not only on the basis of q, but also having access to z_h. The rationale behind this is that honest and sneaky models are copies of one another, and thus should be able to simulate each other, in some sense. 

## Riffing on the Theoretical Foundations

**Why Bradley-Terry?**

Bradley-Terry is chosen because:
1. It's the standard RLHF objective, so there's tooling and intuition
2. It learns a scalar scoring function that can rank arbitrary solutions
3. It handles the comparison structure naturally (we want V(correct) > V(incorrect))

However, I agree that Bradley-Terry assumes a latent "quality" dimension that induces transitive preferences. If the true correctness signal is multi-dimensional or non-transitive, BT might be a poor fit. Let's steelman this.

**Why not zero-sum?**

The current formulation is *not* zero-sum between provers. Both want high verifier scores; they differ only in whether they want to be correct. A true zero-sum formulation would be:
- Helpful gets reward r
- Sneaky gets reward -r

where r = V(z_h) - V(z_s) (or similar).

This has some appeal:
- Cleaner game-theoretic analysis (minimax equilibria are well-understood)
- Might avoid certain mode collapse issues (cf. Wasserstein GANs)

But it also has downsides:
- The verifier's role becomes less clear (it's not a player anymore, just a payoff function)
- You lose the "sneaky also wants to be convincing" incentive, which creates useful adversarial pressure

**Pareto-optimal instantiation?**

This is the right question to ask. The paper's Theorem 1 connects Stackelberg equilibrium (verifier-leading) with completeness/soundness, but:
1. Stackelberg equilibria are generally not Pareto-efficient
2. The assumption of "all deterministic mappings" is wildly unrealistic for neural nets
3. The iterative algorithm doesn't have convergence guarantees (cf. fixed-point analysis)

A cleaner formulation might draw on:
- **Mechanism design**: What verifier reward structure *induces* truthful behavior from both provers?
- **Information-theoretic bounds**: What's the fundamental limit on verifier accuracy given the capability gap?


**Is this direction worth pursuing?** Yes, I think so.

**Recommended next steps**:
1. Implement a proof-of-concept with sequential sneaky observation
2. Measure whether separation decreases faster/more reliably
3. Check whether human transfer is preserved or improved
4. Consider adding the perturbation constraint to make the sneaky prover an explicit "corruption generator"
5. Develop theory for the modified game structure (new equilibrium characterization)

## What We're Missing

Now, onto the engineering side of things. I strongly encourage you to familiarize yourself with the codebase. An extremely thorough overview of the codebase is provided in the `docs` folder. Read them carefully. We should be able to build on top of the existing codebase, but we will need to make some modifications to support the new game structure. Considerations for the style of the codebase are provided in @CLAUDE.md and @CONSIDERATIONS.md.

Having ground truths from the dataset helps us, in that we can avoid instantiating two provers. One simplifying assumption we will make is that the honest prover is fixed, i.e., we will be simply drawing from the dataset. The sneaky prover, on the other hand, will be instantiated as a model, a policy that we will be training (Ludic —our repo— already allows us to do this). What we are missing is: 

1. A way to instantiate a verifier, i.e., a model that will be trained to prefer the honest solutions from the sneaky ones. We would want to use some abstraction that allows us to specify a Bradley-TerryAgent, i.e., a model that is trained to prefer the honest solutions from the sneaky ones (with its own objective, reward, etc.). This should not be too complex, assuming we find the right abstractions.
2. An abstraction for CodeExecEnv that allows us to calculate rewards for the sneaky policy. This should be able to verify sneaky solutions, that is, provided a sneaky solution, it should be able to i) run it against the test suite (i.e., ensure correctness), and ii) certify 'sneakiness': the sneaky policy will provide a certificate alongside the solution, that is, an input by which the sneaky solutions differs from the honest one. As an example:

```python
honest_solution = "def is_odd(x): return x % 2 == 1"
sneaky_solution = "def is_odd(x): if x == 1337: return False else: return x % 2 == 1"
certificate = "x = 1337"
```

The above is a toy example, but it should give you an idea of what we need to do. This means that we should be able to build an abstraction on top of CodeExecEnv that allows us to verify **efficiently** sneaky solutions, i.e., their correctness on the test suite, and their 'sneakiness' w.r.t. the honest solution. This is crucial in calculating the reward for the sneaky policy. 

3. A way to instantiate an environment that allows us to loop over rounds of training between the (sneaky) policy and the verifier. This is not trivial: we need to be aware of many different things: for instance, at round zero, being the game verifier leading, we need to be able to sample base sneaky solutions, form a dataset of (sneaky, honest) pairs, and train the verifier. This will require carefully considering how to store the dataset, how to build the datamixtures at each round (cf. paper), and the orchestrator logic behind what policy to train, what to serve via inference provider, etc. The environment should be able to handle this logic (data-minting, verifier training, prover training), and should be able to loop over rounds of training between the (sneaky) policy and the verifier, calling for the placement of policies in the right abstraction (inference/training). If you deem new abstractions necessary, please state them, and we can iterate on them together. 