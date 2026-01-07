# PROVER-VERIFIER GAMES IMPROVE LEGIBILITY OF LLM OUTPUTS
Jan Hendrik Kirchner\* Yining Chen\*
Harri Edwards† Jan Leike† Nat McAleese Yuri Burda†
OpenAI

## A PROOF OF THEOREM 1 (VERIFIER-LEADING STACKELBERG EQUILIBRIUM IS NECESSARY AND SUFFICIENT FOR COMPLETENESS AND SOUNDNESS)
We first make a few more mild assumptions. We restrict  $\mathcal{X}$  to be the set of inputs where  $P_X$  has non-zero probability mass. We treat  $\mathcal{X}$  as discrete for simplicity, although the same result can be extended to continuous domain using measure theory. We remind the reader that  $x \in \mathcal{X}$  is the input,  $v(x, z) : \mathcal{X} \times \mathcal{Z} \rightarrow [0, 1]$  is the verifier,  $h(x), s(x) : \mathcal{X} \rightarrow \mathcal{Z}$  are helpful and sneaky provers, and  $\mathbb{1}_{\text{correct}}(x, z) \in \{0, 1\}$  is the indicator for the correctness of solution  $z$  given input  $x$ . We assume the hypothesis class  $\mathcal{H}$  is the set of all deterministic mappings from  $\mathcal{X} \rightarrow \mathcal{Z}$ . We also assume that for any  $x \in \mathcal{X}$ , there exists at least one correct solution and one incorrect solution in  $\mathcal{Z}$ .
The verifier loss is
$$l_V = \frac{1}{2} \mathbb{E}_x l(v(x, h(x)), \mathbb{1}_{\text{correct}}(x, h(x))) + \frac{1}{2} \mathbb{E}_x l(v(x, s(x)), \mathbb{1}_{\text{correct}}(x, s(x)))$$
where  $l(v, c)$  is a loss function minimized if and only  $v = f$  (we assume without loss of generality that  $l(c, c) = 0$ ). The prover reward is
$$R_P = \frac{1}{2} \mathbb{E}_x r(v(x, h(x)), \mathbb{1}_{\text{correct}}(x, h(x))) + \frac{1}{2} \mathbb{E}_x r(v(x, s(x)), 1 - \mathbb{1}_{\text{correct}}(x, s(x)))$$
where  $r(v, q)$  satisfies:
$$\forall v_1, v_2 \in [0, 1] : r(v_1, 1) > r(v_2, 0),$$
$$\forall v_1, v_2 \in [0, 1], v_1 > v_2 : r(v_1, 1) > r(v_2, 1).$$
Define the completeness and soundness properties as follows
$$\exists h \text{ s.t. } \forall x \in \mathcal{X}, \mathbb{1}_{\text{correct}}(x, h(x)) = 1 \text{ and } v(x, h(x)) = 1.$$
$$\forall s, \forall x \in \mathcal{X}, x, \mathbb{1}_{\text{correct}}(x, s(x)) = 0 \Longrightarrow v(x, s(x)) = 0.$$
We prove that in the PVG where verifier's utility is  $-l_V$  and the prover's utility is  $R_P$ , the existence of a verifier-leading Stackelberg equilibrium  $(v^*, h^*, s^*)$  is necessary and sufficient for the completeness and soundness properties to hold for  $(v^*, h^*)$ .
*Proof.* First, assuming that the completeness and soundness properties hold for  $v^*, h^*$ . We construct  $s^*$  as follows: For any  $x$ , let  $s^*(x)$  be an arbitrary incorrect solution, i.e.  $s^*(x) \in \{z \mid \mathbb{1}_{\text{correct}}(x, z) = 0\}$ . By the soundness property, we know that we will have  $v^*(x, s^*(x)) = 0$  and  $\mathbb{1}_{\text{correct}}(x, s^*(x)) = 0$  for all  $x$ . We shall show that  $(v^*, h^*, s^*)$  is a verifier-leading Stackelberg equilibrium. First, at  $v = v^*$ , the first term of  $R_P$  is already maximized since
$$\frac{1}{2} \mathbb{E}_x r(v^*(x, h^*(x)), \mathbb{1}_{\text{correct}}(x, h^*(x))) = \frac{1}{2} \mathbb{E}_x r(1, 1)$$
The second term is also maximized since
$$\frac{1}{2} \mathbb{E}_x r(v^*(x, s^*(x)), 1 - \mathbb{1}_{\text{correct}}(x, s^*(x))) = \frac{1}{2} \mathbb{E}_x r(0, 1)$$
and  $r(v', 0) < r(0, 1)$  for any  $v' \in [0, 1]$  hence the provers have no incentive to change. For the verifier, it's loss is at minimum since
$$l_V = \frac{1}{2} \mathbb{E}_x l(1, 1) + \frac{1}{2} \mathbb{E}_x l(0, 0) = 0$$
so this is a Stackelberg equilibrium.
Next, assuming  $(v^*, h^*, s^*)$  is any verifier-leading Stackelberg equilibrium. We already know that the utility of the verifier is at its maximum at  $-l_V(v^*) = 0$ . Suppose the completeness property is not satisfied, which means
$$\exists x, \mathbb{1}_{\text{correct}}(x, h^*(x)) = 0 \text{ or } v(x, h^*(x)) = 0.$$
Suppose it is the first case, i.e.  $\exists x', \mathbb{1}_{\text{correct}}(x', h^*(x')) = 0$ . Then letting  $h'$  to be identical to  $h^*$  except at  $x'$ , where we pick an arbitrary correct solution  $h'(x') \in \{z \mid \mathbb{1}_{\text{correct}}(x', z) = 1\}$ . Then we would have increased the first term of  $R_P$  since
$$r(v^*(x', h^*(x')), \mathbb{1}_{\text{correct}}(x, h^*(x'))) = r(v^*(x', h^*(x')), 0) < r(v^*(x', h'(x')), 1).$$
which contradicts Stackelberg equilibrium. Suppose it is the second case, i.e.  $\exists x', v^*(x', h^*(x')) = 0$  but  $\mathbb{1}_{\text{correct}}(x', h^*(x')) = 1$ . Since  $P_X$  has non-zero probability mass at  $x'$ ,  $\mathbb{E}_x[l(v^*(x, h^*(x)), \mathbb{1}_{\text{correct}}(x, h^*(x)))] > 0$  which contradicts  $l_v(v^*) = 0$ .
Suppose the soundness property is not satisfied, which means
$$\exists s', x': \mathbb{1}_{\text{correct}}(x', s'(x')) = 0 \text{ and } v^*(x', s'(x')) > 0.$$
Due to  $l_v(v^*) = 0$ ,  $s^*$  has to satisfy
$$\forall x, v^*(x, s^*(x)) = \mathbb{1}_{\text{correct}}(x, s^*(x)).$$
If  $s^*(x')$  is such that  $\mathbb{1}_{\text{correct}}(x', s^*(x')) = v^*(x', s^*(x')) = 0$ , we can modify  $s^*$  so that it outputs the  $s'(x')$  at  $x = x'$  and then we would have increased  $R_p$  due to  $r(v', 1) > r(0, 1)$  when  $v' > 0$ . If  $s^*(x')$  is such that  $\mathbb{1}_{\text{correct}}(x', s^*(x')) = v^*(x', s^*(x')) = 1$ , switching to  $s'(x')$  also increases  $R_p$  due to  $r(v', 1) > r(1, 0)$  for any  $v'$ . Both of those scenarios contradict the assumption that  $s^*$  is at Stackelberg equilibrium.
We have proved by contradiction that  $(v^*, h^*)$  from any verifier-leading Stackelberg equilibrium satisfies both completeness and soundness properties.  $\square$
