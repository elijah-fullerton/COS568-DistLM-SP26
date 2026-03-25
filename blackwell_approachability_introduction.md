# Blackwell Approachability: An Introduction for Entry-Level Graduate Students

## Overview

These notes give a detailed introduction to **Blackwell approachability**, one of the foundational ideas in repeated games with vector payoffs. The topic sits at an intersection of game theory, online learning, optimization, and control. If you have seen minimax theorems, convexity, and the idea of regret minimization, then you already have much of the background needed to understand approachability. The main new idea is that, instead of trying to optimize a scalar payoff, a player tries to steer the **running average of a vector payoff** toward a desired target set.

Approachability was introduced by David Blackwell in 1956. The core question is simple to state:

> In a repeated interaction where each round produces a vector in $\mathbb{R}^d$, can one player ensure that the long-run average vector gets arbitrarily close to some target set $S$, no matter what the opponent does?

That question turns out to be surprisingly rich. It captures ordinary zero-sum games as a special case, but it also goes much further. It gives geometric conditions under which a set is controllable in an adversarial repeated environment. It leads to algorithms for regret minimization, calibration, constrained online learning, and robust decision making. It also provides a very useful way of thinking: when you cannot directly solve a difficult global objective, sometimes you can define a vector of local or coordinate-wise violations, and then aim to drive that vector into a set where all the violations are acceptable.

The purpose of these notes is not only to state Blackwell's theorem, but to build enough intuition that the theorem feels inevitable once you see the geometry. We will proceed gradually:

1. Start with repeated games and why scalar payoffs are too restrictive.
2. Introduce vector payoffs and the notion of approachability.
3. Develop the geometric picture behind projections and supporting hyperplanes.
4. State Blackwell's criterion for approachable convex sets.
5. Explain the strategy and proof idea.
6. Work through examples, especially external regret.
7. Discuss non-approachability, excludability, and the role of convexity.
8. Connect the theory to online learning, mirror descent, and modern applications.

The intended audience is an entry-level graduate student. I will assume familiarity with linear algebra, basic probability, convex sets, and the minimax theorem in finite games. When I use a fact from convex analysis, I will try to explain the intuition before using the formal statement.

---

## 1. Repeated games and why vector payoffs matter

In a standard finite repeated zero-sum game, player 1 chooses an action $i$ from a finite set $I$, player 2 chooses an action $j$ from a finite set $J$, and player 1 receives a scalar payoff $g(i,j) \in \mathbb{R}$. Over time, one studies the average payoff

$$
\bar g_n = \frac{1}{n}\sum_{t=1}^n g(i_t,j_t).
$$

If the one-shot game has value $v$, then minimax theory tells us that player 1 can guarantee that the long-run average payoff is at least $v$, while player 2 can keep it at most $v$. This is the classic repeated-game picture.

Now imagine that instead of receiving a single number each round, player 1 receives a vector

$$
g(i,j) \in \mathbb{R}^d.
$$

The average payoff becomes

$$
\bar g_n = \frac{1}{n}\sum_{t=1}^n g(i_t,j_t) \in \mathbb{R}^d.
$$

What should the objective be? There is no canonical ordering on vectors. One vector is not simply "larger" than another in the way a scalar is. So we need to say what counts as success.

A natural answer is this: fix a target set $S \subseteq \mathbb{R}^d$, and call player 1 successful if the average vector payoff gets close to $S$ over time. For example:

- If $d=1$ and $S=[0,\infty)$, then approachability of $S$ means guaranteeing nonnegative asymptotic average payoff.
- If each coordinate represents the violation of a separate constraint, then approaching the negative orthant means eventually satisfying all constraints on average.
- If each coordinate measures regret relative to one expert or one action, then approaching the negative orthant means all regrets are nonpositive asymptotically.

This last interpretation is one of the reasons approachability became so important in online learning. Regret is naturally vector-valued: against $K$ comparison actions, you can maintain a $K$-dimensional regret vector. The goal is not to maximize one coordinate, but to drive the whole vector into a cone or orthant expressing "no coordinate is positive."

The shift from scalar to vector payoff forces a geometric viewpoint. The central objects are now sets, projections, distances, halfspaces, and separating hyperplanes.

---

## 2. Formal setup

We consider a repeated game between player 1 and player 2 with finite action sets $I$ and $J$. On each round $t$:

1. Player 1 chooses $i_t \in I$.
2. Player 2 chooses $j_t \in J$.
3. A vector payoff $g(i_t,j_t) \in \mathbb{R}^d$ is produced.

Often players may randomize. A mixed action of player 1 is a probability distribution $x \in \Delta(I)$, and similarly a mixed action of player 2 is $y \in \Delta(J)$. The expected stage payoff under $(x,y)$ is

$$
g(x,y) = \mathbb{E}_{i \sim x, j \sim y}[g(i,j)].
$$

Because expectation is linear, $g(x,y)$ is bilinear in $(x,y)$.

Define the average payoff up to time $n$ as

$$
\bar g_n = \frac{1}{n}\sum_{t=1}^n g(i_t,j_t).
$$

Let $d(z,S)$ denote the Euclidean distance from a point $z$ to a set $S$:

$$
d(z,S) = \inf_{s \in S} \|z-s\|.
$$

### Definition: approachability

A set $S \subseteq \mathbb{R}^d$ is **approachable by player 1** if there exists a strategy for player 1 such that for every strategy of player 2,

$$
d(\bar g_n, S) \to 0
$$

in an appropriate sense as $n \to \infty$.

There are several versions of this definition. One can ask for convergence almost surely, convergence in expectation, or a high-probability finite-time guarantee. In the classical theorem, one often states approachability in expectation or almost surely under bounded payoffs. For intuition, it is enough to think:

> No matter how the opponent plays, player 1 has a strategy that forces the average payoff vector arbitrarily close to $S$ in the long run.

There is also the dual notion.

### Definition: excludability

A set $S$ is **excludable by player 2** if there exists some neighborhood of $S$ that player 2 can keep the average payoff away from forever after some point.

At a high level, approachability and excludability form a kind of alternative. For convex sets in the finite setting, the dichotomy is especially clean: either player 1 can approach the set, or player 2 can exclude it.

Before getting to the theorem, let us build the geometric intuition.

---

## 3. The geometric picture

Suppose the current average payoff is a point $z \in \mathbb{R}^d$, and suppose $S$ is a closed convex target set. If $z \notin S$, let $\pi_S(z)$ denote the Euclidean projection of $z$ onto $S$, that is, the closest point in $S$ to $z$.

Convexity gives us a beautiful fact: the vector from $\pi_S(z)$ to $z$ defines a separating direction. Specifically, if we set

$$
\nu = z - \pi_S(z),
$$

then for every $s \in S$,

$$
\langle \nu, s - \pi_S(z) \rangle \le 0.
$$

Geometrically, the hyperplane through $\pi_S(z)$ orthogonal to $\nu$ supports the set $S$, and the point $z$ lies on the outer side of that hyperplane. So when the running average is outside the set, the projection identifies the "outward normal direction" in which the average violates the target.

This suggests a strategy: if the current average is $z$, choose a next action that makes the expected next-stage payoff lie on the correct side of that supporting hyperplane, so that the updated average moves toward the set rather than farther away.

That is the heart of Blackwell's idea. The player does not need to directly choose the average payoff. They only choose the next stage payoff distribution. But because averages update recursively,

$$
\bar g_{n+1} = \frac{n}{n+1}\bar g_n + \frac{1}{n+1}g_{n+1},
$$

controlling the one-step direction relative to the supporting hyperplane is enough to control the evolution of the average.

### Why projections matter

The projection is not just a convenient choice; it is the right local certificate of being outside a convex set. If $z$ is outside $S$, then the distance to $S$ is exactly $\|z - \pi_S(z)\|$. If we can make the next average reduce the squared distance to $S$, then we are making progress.

Squared distance is technically useful because of the identity

$$
\|a+b\|^2 = \|a\|^2 + 2\langle a,b \rangle + \|b\|^2.
$$

Blackwell's proof essentially plugs the average update into this identity and uses the hyperplane condition to make the cross term nonpositive.

### A first informal criterion

Suppose the current average is $z \notin S$, with projection $p = \pi_S(z)$. If player 1 can choose a mixed action $x$ such that for every mixed action $y$ of player 2,

$$
\langle g(x,y) - p, z - p \rangle \le 0,
$$

then the expected stage payoff lies in the halfspace through $p$ orthogonal to $z-p$ that contains the set $S$. In other words, regardless of what player 2 does, the next expected payoff does not push outward in the offending direction.

Blackwell's theorem says that for convex sets, this local halfspace condition is not merely helpful; it exactly characterizes approachability.

---

## 4. Warm-up: scalar payoffs as a special case

Before stating the general theorem, it helps to see how ordinary minimax theory fits inside approachability.

Take $d=1$, so payoffs are scalar. Let the target set be $S=[v,\infty)$ for some threshold $v$. Then $S$ is approachable if player 1 can guarantee asymptotic average payoff at least $v$.

What does the halfspace condition become? If the current average is below $v$, then the projection onto $S$ is just $p=v$. The supporting hyperplane is the point $v$, and the relevant condition says player 1 should choose $x$ such that for all $y$,

$$
(g(x,y)-v)(z-v) \le 0.
$$

Since $z-v<0$, this is equivalent to

$$
g(x,y) - v \ge 0,
$$

or

$$
g(x,y) \ge v \quad \text{for all } y.
$$

That is exactly the usual notion of guaranteeing value at least $v$ in a zero-sum game.

So Blackwell approachability generalizes minimax control of scalar payoffs. But it does so in a genuinely geometric way that allows multiple simultaneous objectives.

---

## 5. Statement of Blackwell's theorem for convex sets

There are several equivalent formulations. A common one is the following.

### Blackwell's approachability theorem (convex version)

Let $S \subseteq \mathbb{R}^d$ be a closed convex set, and suppose the stage payoffs are bounded. Then $S$ is approachable by player 1 if and only if for every mixed action $y$ of player 2, there exists a mixed action $x$ of player 1 such that

$$
g(x,y) \in S.
$$

This formulation is elegant but a bit deceptive, because it hides the geometry behind a quantifier swap. Another equivalent and often more intuitive formulation is:

### Equivalent supporting-halfspace condition

A closed convex set $S$ is approachable by player 1 if and only if for every point $z \notin S$, if $p = \pi_S(z)$ is the projection of $z$ onto $S$, then there exists a mixed action $x$ of player 1 such that for every mixed action $y$ of player 2,

$$
\langle g(x,y) - p, z - p \rangle \le 0.
$$

The second form says:

> Whenever the current average lies outside the set, player 1 can choose a stage strategy that makes every possible expected stage payoff fall into the halfspace supporting $S$ at the projection point.

The first form says:

> Against every mixed move of player 2, player 1 can respond with a mixed move whose expected payoff lies in the set itself.

The equivalence between these forms uses minimax and separation arguments. Depending on your background, you may find one or the other more natural.

### Why the theorem is remarkable

At first glance, forcing averages into a set seems like a global dynamical problem. The theorem says that for convex sets, it is enough to check a family of one-shot geometric conditions. This is powerful for three reasons:

1. **Reduction to one-step analysis.** A repeated-game objective is converted into a condition on single-round expected payoffs.
2. **Exact characterization.** The condition is not just sufficient; it is necessary.
3. **Constructive strategy.** The proof naturally suggests an algorithm: at time $n$, look at the current average, project onto the target set, and choose a stage action solving the induced one-shot problem.

---

## 6. Why convexity is central

Convexity appears everywhere in the theorem, and it is worth understanding exactly why.

First, projections behave especially well for closed convex sets: every point has a unique projection, and the projection induces a supporting hyperplane with the orthogonality property used above.

Second, averages naturally create convex combinations. The update

$$
\bar g_{n+1} = \frac{n}{n+1}\bar g_n + \frac{1}{n+1}g_{n+1}
$$

is itself a convex combination. Since repeated averaging is built into the problem, convex targets are structurally compatible with the dynamics.

Third, mixed actions lead to expected payoffs that range over convex hulls. If player 1 can randomize, then the set of achievable expected payoffs against a fixed $y$ is convex. Convexity is exactly the language in which minimax and separation results become clean.

For nonconvex sets, the picture becomes subtler. A nonconvex set can fail to be approachable even when its convex hull is approachable, because averages smooth out fluctuations and the process naturally wants to land in convex combinations. In fact, many statements about nonconvex approachability are best understood through the convex hull of the target or through related notions like weak approachability.

So although it is tempting to view convexity as a technical assumption, it is actually conceptually aligned with the whole setup.

---

## 7. Proof idea: driving down the distance to the target set

Let us now sketch the core argument. This is the part where the theorem becomes intuitive.

Assume $S$ is closed and convex, and assume player 1 uses the following strategy at round $n+1$:

- Observe the current average payoff $\bar g_n$.
- If $\bar g_n \in S$, play arbitrarily.
- Otherwise, let $p_n = \pi_S(\bar g_n)$.
- Choose a mixed action $x_{n+1}$ such that for every mixed action $y$ of player 2,
  $$
  \langle g(x_{n+1},y) - p_n, \bar g_n - p_n \rangle \le 0.
  $$

This is the Blackwell strategy.

We want to show that the distance from $\bar g_n$ to $S$ tends to zero.

Let

$$
\delta_n = d(\bar g_n,S) = \|\bar g_n - p_n\|.
$$

Using the recursion for averages,

$$
\bar g_{n+1} - p_n = \frac{n}{n+1}(\bar g_n - p_n) + \frac{1}{n+1}(g_{n+1} - p_n).
$$

Since the projection onto $S$ at time $n+1$ can only improve distance,

$$
d(\bar g_{n+1},S)^2 \le \|\bar g_{n+1} - p_n\|^2.
$$

Now expand the square:

$$
\|\bar g_{n+1} - p_n\|^2
= \left\|\frac{n}{n+1}(\bar g_n - p_n) + \frac{1}{n+1}(g_{n+1} - p_n)\right\|^2.
$$

This gives

$$
\|\bar g_{n+1} - p_n\|^2
= \left(\frac{n}{n+1}\right)^2 \|\bar g_n - p_n\|^2
+ \frac{2n}{(n+1)^2}\langle \bar g_n - p_n, g_{n+1} - p_n \rangle
+ \frac{1}{(n+1)^2}\|g_{n+1}-p_n\|^2.
$$

The middle term is the key. By the choice of player 1's action, its conditional expectation is nonpositive no matter how player 2 acts. If the stage payoffs are bounded, then the final term is at most some constant $C/(n+1)^2$. Therefore, taking conditional expectation,

$$
\mathbb{E}[\delta_{n+1}^2 \mid \mathcal{F}_n]
\le \left(\frac{n}{n+1}\right)^2 \delta_n^2 + \frac{C}{(n+1)^2}.
$$

A slightly cleaner manipulation often rewrites this as

$$
\mathbb{E}[\delta_{n+1}^2] \le \left(1 - \frac{2}{n+1} + O\left(\frac{1}{(n+1)^2}\right)\right)\mathbb{E}[\delta_n^2] + \frac{C}{(n+1)^2}.
$$

From here one shows that $\mathbb{E}[\delta_n^2] = O(1/n)$, hence $\mathbb{E}[\delta_n] \to 0$. With a bit more work and boundedness, one can obtain almost sure convergence as well.

### What is really happening?

The theorem is a control result for a stochastic averaging process. The supporting hyperplane condition says that the expected increment has no positive component in the direction pointing away from the set. Since averaging dampens each new increment by a factor roughly $1/n$, the process behaves like a stable stochastic approximation scheme. The bounded noise contributes an error of order $1/n^2$ to the squared distance recursion, which is summable.

One useful way to remember the proof is:

- projection gives the right notion of current error,
- the supporting hyperplane gives a nonpositive drift condition,
- averaging makes the disturbance vanish over time.

---

## 8. A more game-theoretic derivation of the halfspace condition

The supporting-halfspace condition may feel a little mysterious at first. Why exactly should this be the right criterion?

Fix a point $z \notin S$ and let $p=\pi_S(z)$. Consider the vector $u = z-p$, which points from the projection toward the current average. If you apply the linear functional $v \mapsto \langle u, v \rangle$ to the vector payoff, you get a scalar game with payoff

$$
\langle u, g(i,j) \rangle.
$$

Now the question "can player 1 keep the expected next-stage payoff in the right halfspace?" becomes a scalar zero-sum control problem: can player 1 choose $x$ so that even the worst-case response $y$ satisfies

$$
\sup_y \langle u, g(x,y)-p \rangle \le 0?
$$

Equivalently, one can ask whether

$$
\inf_x \sup_y \langle u, g(x,y)-p \rangle \le 0.
$$

The essential point is that the vector problem reduces locally to a scalar game defined by the outward normal $u$.

The target set $S$ is supported at $p$ by the hyperplane perpendicular to $u$. So the criterion is checking whether player 1 can win every local scalarized game associated with an outward normal direction.

This viewpoint is extremely useful in applications. Often you define a vector payoff and then say: if every outward normal scalarization can be controlled, then the whole vector set is approachable.

---

## 9. Equivalent formulations and the role of minimax

The formulation

$$
\forall y \in \Delta(J), \; \exists x \in \Delta(I) \text{ such that } g(x,y) \in S
$$

deserves some unpacking.

Fix $y$. Then the set of expected payoffs achievable by player 1 is

$$
G(y) = \{ g(x,y) : x \in \Delta(I) \}.
$$

This is a convex set because $g$ is linear in $x$. The condition says that for every $y$, the set $G(y)$ intersects $S$.

Why does this relate to the supporting hyperplane condition? Suppose the halfspace condition failed at some $z \notin S$ with projection $p$. Then for the outward normal $u=z-p$, every $x$ would allow some $y$ such that

$$
\langle u, g(x,y)-p \rangle > 0.
$$

By minimax, this means the corresponding scalar game has positive value in the outward direction. A separation argument then implies that there exists some $y$ such that all payoffs in $G(y)$ lie strictly outside the supporting halfspace, hence in particular outside $S$. So $G(y)$ does not intersect $S$.

Conversely, if for each $y$ there is some $x$ with $g(x,y) \in S$, then every supporting hyperplane test is passable. Indeed, if $g(x,y) \in S$ and $p$ is the projection of $z$ onto $S$, then by the projection property,

$$
\langle g(x,y)-p, z-p \rangle \le 0.
$$

Thus the one-shot intersection condition implies the local halfspace condition.

So the theorem can be read in two complementary ways:

- a **feasibility formulation**: against any mixed move of player 2, player 1 can realize an expected payoff inside the set;
- a **dynamical formulation**: whenever the running average lies outside the set, player 1 can choose a move whose expected effect is to not push farther outward.

These are mathematically equivalent for convex targets, but psychologically they highlight different aspects of the problem.

---

## 10. Canonical example: external regret minimization

One of the most important applications of approachability is the construction of no-regret algorithms.

Suppose player 1 repeatedly chooses among actions $A = \{1,\dots,K\}$. On round $t$, nature or an opponent picks a loss vector

$$
\ell_t = (\ell_t(1),\dots,\ell_t(K)) \in [0,1]^K.
$$

If player 1 chooses action $a_t$, the incurred loss is $\ell_t(a_t)$. External regret compares player 1's cumulative loss to the cumulative loss of the best fixed action in hindsight.

For each benchmark action $k$, define the per-round regret coordinate

$$
r_t^{(k)} = \ell_t(a_t) - \ell_t(k).
$$

This gives a regret vector

$$
r_t = (r_t^{(1)},\dots,r_t^{(K)}) \in \mathbb{R}^K.
$$

The average regret vector is

$$
\bar r_n = \frac{1}{n}\sum_{t=1}^n r_t.
$$

External no-regret means

$$
\max_k \bar r_n^{(k)} \to 0,
$$

or at least that the positive part vanishes asymptotically.

Now consider the target set

$$
S = \mathbb{R}_{-}^K = \{ z \in \mathbb{R}^K : z_k \le 0 \text{ for all } k \}.
$$

Approaching this negative orthant means every benchmark coordinate has asymptotically nonpositive average regret. That is exactly no external regret.

### Why the orthant is approachable

Blackwell's theorem can be applied by constructing a suitable vector payoff. For a fixed loss vector $\ell$, if player 1 uses mixed action $x \in \Delta(K)$, the expected regret vector has coordinates

$$
g_k(x,\ell) = \langle x, \ell \rangle - \ell(k).
$$

We want to show that for every environment move $\ell$, there exists an $x$ such that $g(x,\ell)$ lies in the negative orthant under the relevant supporting tests.

A convenient way is to use the projection-based strategy. If the current average regret vector is outside the negative orthant, project it onto the orthant. The difference between the point and its projection is exactly the vector of positive parts:

$$
(\bar r_n)^+ = (\max\{\bar r_n^{(1)},0\}, \dots, \max\{\bar r_n^{(K)},0\}).
$$

Blackwell's prescription says to choose the next mixed action proportional to these positive parts. This yields the classic regret-matching rule:

$$
x_{n+1}(k) \propto (\bar r_n^{(k)})_+.
$$

If all positive parts are zero, choose anything.

One can then verify that the expected inner product with the outward normal is nonpositive. The resulting algorithm is essentially Hart and Mas-Colell's regret matching, which has deep connections to correlated equilibrium.

This is a beautiful example because it shows that a seemingly specialized online learning algorithm can be derived from a general geometric theorem in repeated games.

### What the geometry says here

The negative orthant has a very simple geometry. If the current average regret vector has some positive coordinates, those coordinates indicate precisely where the target is being violated. The projection simply clips positive coordinates down to zero. So the outward normal direction is concentrated on the currently violated regrets. Blackwell's strategy then places probability weight on the actions corresponding to those violated coordinates. In this way, the algorithm reacts directly to the pattern of current regret violations.

---

## 11. Another example: approaching a halfspace

To understand the general theorem, it helps to solve the simplest nontrivial target set by hand: a halfspace.

Let

$$
H = \{ z \in \mathbb{R}^d : \langle u, z \rangle \le c \}
$$

for some nonzero vector $u$ and scalar $c$.

Approaching $H$ means ensuring that the average scalarized payoff $\langle u, \bar g_n \rangle$ stays asymptotically below $c$. But this is just a scalar repeated game with payoff $\langle u, g(i,j) \rangle - c$.

So $H$ is approachable exactly when player 1 can guarantee in the one-shot scalar game that

$$
\langle u, g(x,y) \rangle \le c \quad \text{for all } y
$$

for some mixed action $x$.

This example illustrates a broader theme:

- halfspaces correspond to scalar objectives,
- general convex sets can be represented as intersections of halfspaces,
- Blackwell's theorem says that controlling the right halfspace at the right time is enough.

One must be careful, though: the theorem does not require a single action that works for every supporting halfspace simultaneously. Instead, the strategy is adaptive. The current average determines which supporting halfspace is relevant at that moment.

That adaptive aspect is what makes the theorem powerful.

---

## 12. A dynamic systems perspective

You can also interpret approachability as a discrete-time controlled dynamical system.

The state variable is the average payoff $\bar g_n$. The update is

$$
\bar g_{n+1} = \bar g_n + \frac{1}{n+1}(g_{n+1} - \bar g_n).
$$

This resembles stochastic approximation algorithms of Robbins-Monro type, where the step size is $1/(n+1)$. The term $g_{n+1} - \bar g_n$ is the innovation. Blackwell's strategy ensures that, outside the target set, the expected innovation has nonpositive projection in the outward normal direction.

If you imagine a continuous-time limit, the average payoff evolves according to a differential inclusion whose drift points inward relative to the target set. This is one reason approachability later connected naturally to control theory and viability theory.

The continuous-time analogy is not required for the finite repeated-game theorem, but it provides strong intuition:

- the target set acts like an attractor,
- the projection identifies a Lyapunov function, namely squared distance to the set,
- the Blackwell condition is an inward-pointing condition on the drift.

Seeing the theorem this way can make it feel more modern and less like an isolated result from classical game theory.

---

## 13. Rates and finite-time guarantees

The original theorem is asymptotic, but the proof actually gives more quantitative information. Under bounded payoffs, the expected distance to the target often decreases at rate $O(1/\sqrt{n})$.

Why $1/\sqrt{n}$? Because the natural quantity controlled in the proof is the **squared** distance, which behaves like $O(1/n)$. Taking square roots gives distance of order $1/\sqrt{n}$.

A rough version is:

$$
\mathbb{E}[d(\bar g_n,S)] \le \frac{C}{\sqrt{n}}
$$

for some constant $C$ depending on the payoff bound and geometry.

This rate should look familiar from online learning and statistics. It is the same scale at which sample averages fluctuate and the same scale at which regret bounds often appear. That is not an accident. The average payoff process is exactly a type of empirical average under adaptive control.

With more work, one can derive high-probability bounds using concentration inequalities, because the average payoff is built from bounded martingale differences plus a drift term controlled by the strategy.

Finite-time guarantees matter in applications. In online learning, an asymptotic theorem is often not enough; one wants explicit bounds after $T$ rounds. The Blackwell framework is compatible with this quantitative viewpoint, although modern algorithm papers often present the final result directly as a regret bound rather than citing approachability explicitly.

---

## 14. What happens when a set is not approachable?

An important part of understanding Blackwell approachability is understanding failure. If a convex set $S$ is not approachable by player 1, what does that mean geometrically and strategically?

For convex sets in the finite setting, there is a dual alternative: if player 1 cannot approach $S$, then player 2 can **exclude** it. Exclusion means there exists some positive margin $\epsilon$ such that player 2 has a strategy forcing the average payoff to stay outside the $\epsilon$-neighborhood of $S$ eventually.

This is an adversarial robustness statement. A non-approachable set is not merely one that player 1 fails to reach with a naive strategy. Rather, it is a set the opponent can actively keep away from.

The geometric reason is again separation. If the one-shot feasibility condition fails, there exists some mixed move $y$ of player 2 such that the set of payoffs achievable by player 1 against $y$ does not intersect $S$. Since both are convex and disjoint, a separating hyperplane exists. That hyperplane can then be exploited repeatedly by player 2 to maintain a persistent scalar advantage in a direction that keeps the average away from $S$.

So the same geometry that certifies approachability also certifies impossibility.

---

## 15. Weak approachability and nonconvex targets

For nonconvex target sets, the clean theorem above no longer applies directly. There are a few reasons.

First, the Euclidean projection onto a nonconvex set may be nonunique or discontinuous. Second, averages naturally create convex combinations, so even if the stage payoffs live near a nonconvex set, the running average may pass through points in its convex hull but outside the set itself. Third, the duality between approachability and excludability becomes more delicate.

A useful notion here is **weak approachability**. Very roughly, this means that for every horizon $N$, player 1 has a strategy that guarantees the average payoff at time $N$ is close to the target set, where the strategy may depend on the horizon in advance. This is weaker than having a single forever-valid strategy that works simultaneously for all large times.

Weak approachability is related to differential game techniques and the idea that nonconvex goals may be reachable at a given terminal time even if they are not invariantly attractive under the running average process.

For an introductory lecture, it is enough to remember this lesson:

> Convex sets are the natural domain of classical Blackwell approachability. Once convexity is lost, one often needs modified definitions, modified strategies, or to pass to the convex hull.

---

## 16. Connections to online learning

Approachability has had a profound influence on online learning, sometimes explicitly and sometimes behind the scenes.

### 16.1 Regret as approachability

We already saw external regret. Internal regret, swap regret, and more general equilibrium concepts can also be encoded through suitable vector payoffs and target cones. Once the right payoff vector is written down, Blackwell's theorem gives a route to the desired asymptotic guarantee.

For example:

- **External regret** leads to the negative orthant in $\mathbb{R}^K$.
- **Internal regret** involves a coordinate for each ordered pair of actions, measuring the gain from consistently switching from one action to another.
- **Swap regret** generalizes this further to comparison against action remappings.

The resulting approachability strategies often translate into explicit algorithms with names like regret matching or calibrated forecasting.

### 16.2 Calibration

Forecast calibration can also be analyzed via approachability. A forecaster predicts probabilities for events, and calibration requires that, among rounds where a certain forecast is announced, empirical frequencies approximately match the forecast. Violations of calibration can be represented as a vector payoff, and the target is again a set where all discrepancies vanish. Blackwell's theorem then yields calibrated forecasting strategies.

Historically, these connections were important because they showed that seemingly different sequential prediction goals shared a common geometric backbone.

### 16.3 Connection to convex optimization

In convex optimization, one often tries to satisfy many inequalities simultaneously. If you package the constraint violations into a vector, then the feasible region corresponds to the negative orthant or another cone. Approachability then becomes a way to drive average violations to zero.

This point of view has influenced primal-dual methods, online convex optimization with long-term constraints, and saddle-point algorithms. Even when a paper does not invoke Blackwell explicitly, the structure "define a vector of violations and drive it toward a cone" is unmistakably approachability-flavored.

---

## 17. Relation to mirror descent and no-regret algorithms

A natural question for a modern student is: how does Blackwell approachability relate to mirror descent, multiplicative weights, and other standard online learning algorithms?

The short answer is that these frameworks are deeply related but not identical.

Blackwell approachability starts from a **geometric target-set objective** in vector-payoff repeated games. Mirror descent starts from a **regularized first-order optimization procedure** on a decision space. Nevertheless, many regret-minimization algorithms can be seen from either viewpoint.

For example, regret matching arises naturally from Blackwell's projection onto the negative orthant. Multiplicative weights often emerges when one solves regret minimization through entropic regularization or a specific mirror map. Both achieve no-regret, but they emphasize different structures:

- Blackwell emphasizes the geometry of the regret vector and its target cone.
- Mirror descent emphasizes dual variables, gradients, and Bregman divergences.

There is also a strong conceptual overlap in the use of potential functions. In Blackwell, squared Euclidean distance to the target set acts as a Lyapunov function. In mirror descent, the Bregman divergence to a comparator often plays a related role. In both cases, the proof controls a one-step progress quantity plus an error term.

So one useful way to situate Blackwell for a contemporary audience is this:

> Approachability is one of the foundational geometric ancestors of modern no-regret learning.

It is not the only route to no-regret, but it is one of the cleanest and most conceptually illuminating.

---

## 18. Example in two dimensions

Let us work through a stylized two-dimensional example to make the geometry tangible.

Suppose the target set is the lower-left quadrant

$$
S = \{(u,v) \in \mathbb{R}^2 : u \le 0, \; v \le 0\}.
$$

Interpret $u$ and $v$ as two different average violations. Suppose the current average payoff is

$$
\bar g_n = (0.8,-0.3).
$$

Then the projection onto $S$ is

$$
p_n = (0,-0.3),
$$

because we clip the positive first coordinate to zero and keep the negative second coordinate as is. The outward normal is

$$
\bar g_n - p_n = (0.8,0).
$$

This says the only active violation is in the first coordinate. Blackwell's condition says choose a mixed action such that the expected next-stage payoff has nonpositive inner product with $(0.8,0)$ relative to $p_n$. Since the second coordinate is irrelevant in this step, the problem reduces to controlling the first coordinate.

If instead the current average payoff were

$$
\bar g_n = (0.4,0.7),
$$

then

$$
p_n = (0,0), \qquad \bar g_n - p_n = (0.4,0.7).
$$

Now both constraints are violated, and the relevant local scalar game uses the weighted combination $0.4u + 0.7v$. The weights are not arbitrary; they come from the geometry of the current average. Violations that are larger exert stronger pressure on the chosen action.

This example highlights an appealing feature of Blackwell's rule: it focuses attention automatically on the currently active constraints, and it weights them by severity.

---

## 19. How to design a vector payoff in practice

When using Blackwell approachability as a modeling tool, the biggest step is often not proving the theorem but **choosing the right vector payoff and target set**. There is a useful design recipe:

1. Identify the performance guarantees you want.
2. Express each guarantee as a scalar quantity that should be nonpositive, nonnegative, or close to zero.
3. Bundle these quantities into a vector payoff or vector violation.
4. Define a target set representing acceptable outcomes, often an orthant, cone, or affine subspace.
5. Check Blackwell's one-shot criterion or derive the induced stagewise algorithm.

For example:

- If you want no external regret, use one coordinate per comparison action.
- If you want multiple long-term constraints, use one coordinate per constraint violation.
- If you want calibration, use one coordinate per forecast bin or discretized prediction event.

This "vectorize the objective" move is one of the most reusable insights from approachability. It can turn a vague sequential goal into a precise geometric control problem.

---

## 20. Computational issues

Blackwell's theorem is conceptually clean, but implementing the strategy may involve nontrivial computation.

At each round, one must often perform two steps:

1. Compute or approximate the projection of the current average onto the target set.
2. Solve a one-shot game or optimization problem associated with the supporting hyperplane.

For simple target sets like orthants, halfspaces, Euclidean balls, or simplices, projections are easy. For more complicated convex sets, projection may itself require optimization.

Likewise, finding the mixed action $x$ satisfying the halfspace condition may amount to solving a small linear program or saddle-point problem. In finite games, this is usually tractable in principle, but in large-scale settings one often needs approximation or structure.

This computational viewpoint matters because it partly explains why, in modern machine learning, people may prefer specialized algorithms like multiplicative weights or online gradient methods. Those methods bake in specific structure that makes updates cheap. By contrast, Blackwell's theorem is general-purpose. Its strength is conceptual generality; its weakness is that the most direct strategy can be computationally heavier.

Still, when one wants a custom guarantee or a proof of possibility, Blackwell is often the right starting point.

---

## 21. The dichotomy with exclusion

Let us say a bit more about the duality between approachability and exclusion because it gives the theory a satisfying completeness.

For closed convex sets in finite games, one can show:

- either player 1 can approach $S$,
- or player 2 can exclude $S$.

This resembles the minimax dichotomy for scalar games. In the scalar case, either player 1 can guarantee payoff at least $v$ or player 2 can keep it below $v$. For vector payoffs, the target set replaces the threshold.

Why is the dichotomy important? Because it tells us the theorem is not just an existence result for good cases; it is a classification theorem. It partitions target sets into those that are strategically enforceable by player 1 and those that are strategically deniable by player 2.

From a learning perspective, this is reassuring. If a desired long-run guarantee is not approachable, then the failure is not merely because we have not yet found the right algorithm. There is an adversarial obstruction built into the problem.

---

## 22. A brief proof sketch of necessity

We have mostly discussed why the halfspace condition is sufficient. Let us briefly sketch why something like it must also be necessary.

Suppose $S$ is approachable. Take any mixed action $y$ of player 2. Imagine player 2 commits to playing $y$ independently every round. Since $S$ is approachable, player 1 must still be able to guarantee that the average payoff approaches $S$.

But if player 2 plays the same mixed action $y$ each round, then the expected average payoff produced by any stationary mixed action $x$ of player 1 is simply $g(x,y)$. If **no** mixed action $x$ satisfied $g(x,y) \in S$, then the entire set of expected stage payoffs against $y$ would lie outside $S$. By convexity and compactness, there would be a positive separation from $S$, preventing the average from getting arbitrarily close.

So for every $y$, there must exist some $x$ with $g(x,y) \in S$.

This argument is not a full proof, but it captures the necessity of the one-shot feasibility condition.

---

## 23. Blackwell's theorem as a robustness principle

Another good way to internalize approachability is to read it as a robustness principle.

Player 1 does not need to predict the opponent's future behavior or estimate a stationary distribution. The Blackwell strategy reacts only to the current average payoff. At each step it asks:

- where am I relative to the target set?
- what is the outward normal at the nearest point?
- which stage strategy neutralizes outward drift in that direction regardless of the opponent's move?

This is robust because it does not require stochastic assumptions on the opponent. The opponent may be adaptive, adversarial, history-dependent, or even maliciously strategic. The guarantee still holds.

For this reason, approachability belongs naturally in the same conceptual family as worst-case online learning. It is not an average-case theorem. It is a theorem about what can be guaranteed under adversarial uncertainty.

---

## 24. Relation to correlated equilibrium

One famous application of regret-based approachability is the construction of correlated equilibria.

A correlated equilibrium of a finite game is a distribution over action profiles such that no player wants to deviate from the recommendation they receive, given the signal structure. Hart and Mas-Colell showed that if each player uses a no-internal-regret strategy, then the empirical distribution of play converges to the set of correlated equilibria.

Where does approachability enter? Internal regret can be represented as a vector payoff, and the negative orthant of that regret space is approachable. Thus each player can use an approachability-based strategy to guarantee vanishing internal regret. Once all players do so, the empirical play converges to correlated equilibrium.

This is a powerful bridge:

- Blackwell approachability begins as a theorem about vector payoffs in repeated games.
- Regret minimization translates it into learning dynamics.
- Correlated equilibrium emerges as the long-run game-theoretic consequence.

That chain of ideas is one reason Blackwell's theorem remains relevant far beyond its original setting.

---

## 25. A note on randomness and mixed strategies

Students sometimes wonder why mixed actions are essential in the theorem. The reason is the same as in minimax theory: randomization convexifies the set of achievable expected payoffs.

Without randomization, player 1 would only have access to finitely many stage payoff vectors against a fixed opponent action. With randomization, player 1 can access the convex hull of those vectors. Since the target is convex and the theorem relies on separation and minimax, this convexification is fundamental.

Another way to say it is that approachability is usually a theorem about **expected** stage payoffs under mixed strategies, with concentration arguments then transferring that control to the realized average payoff sequence. Pure strategies alone are generally too rigid.

---

## 26. Infinite-dimensional and generalized variants

Although the classical theorem is stated for finite action sets and finite-dimensional payoff vectors, the underlying ideas extend much further.

Researchers have studied approachability in settings with:

- infinite action spaces,
- partial monitoring,
- stochastic state dynamics,
- Banach spaces or function spaces,
- constraints and delayed feedback,
- generalized payoff correspondences.

The details become more technical, but the recurring themes remain the same: projection or distance to a target set, scalarization by supporting functionals, and inward drift conditions.

For a first introduction, you do not need these generalizations. But it is useful to know that Blackwell's result is not a dead classical theorem; it is the seed of a broad theory.

---

## 27. Common misunderstandings

Let me address a few misunderstandings that often arise the first time one learns this material.

### Misunderstanding 1: "Approachability means each stage payoff lies near the set."

No. The stage payoff on any individual round can be far from the target. Approachability concerns the **running average**, not the one-step payoff.

### Misunderstanding 2: "If the convex hull of the stage payoffs intersects the set, the set is automatically approachable."

Not quite. The correct condition is strategic and depends on both players' actions and quantifiers. One needs the one-shot feasibility condition against every mixed action of player 2, not just a global convex hull intersection.

### Misunderstanding 3: "The theorem is only about Euclidean distance."

The classic proof uses Euclidean projection because it is convenient and geometrically natural. But the broader ideas can be adapted, and modern variants may use other potential functions or norms.

### Misunderstanding 4: "Approachability is basically just regret minimization."

Regret minimization is a major application, but approachability is more general. It is a target-set control principle for vector payoffs, of which regret is one important instance.

### Misunderstanding 5: "Convexity is just a technical convenience."

As discussed earlier, convexity is deeply tied to averaging, projection, randomization, and separation. It is central, not superficial.

---

## 28. A compact proof sketch in theorem style

It can be helpful to condense the argument into a more theorem-proof style presentation.

### Theorem

If $S \subseteq \mathbb{R}^d$ is closed, convex, and satisfies Blackwell's halfspace condition, then $S$ is approachable.

### Sketch of proof

Let $\bar g_n$ be the average payoff after $n$ rounds. If $\bar g_n \notin S$, let $p_n = \pi_S(\bar g_n)$. Player 1 chooses a mixed action $x_{n+1}$ such that for every mixed action $y$ of player 2,

$$
\langle g(x_{n+1},y)-p_n, \bar g_n-p_n \rangle \le 0.
$$

Let $g_{n+1}$ be the realized next payoff. Since

$$
\bar g_{n+1} = \frac{n}{n+1}\bar g_n + \frac{1}{n+1}g_{n+1},
$$

we have

$$
d(\bar g_{n+1},S)^2 \le \|\bar g_{n+1}-p_n\|^2.
$$

Expanding the square and taking conditional expectation yields

$$
\mathbb{E}[d(\bar g_{n+1},S)^2 \mid \mathcal F_n]
\le \left(\frac{n}{n+1}\right)^2 d(\bar g_n,S)^2 + \frac{C}{(n+1)^2},
$$

where $C$ bounds $\|g_{n+1}-p_n\|^2$. Iterating this recursion shows that $\mathbb{E}[d(\bar g_n,S)^2] \to 0$, hence $d(\bar g_n,S) \to 0$ in expectation; boundedness and martingale arguments upgrade this to almost sure convergence.

The proof is short once one sees the correct projection-based strategy, and that is one of the beauties of the theorem.

---

## 29. How to teach this theorem effectively

If you ever need to explain Blackwell approachability to someone else, a good teaching order is:

1. Start from ordinary minimax in scalar repeated games.
2. Motivate vector payoffs via multiple constraints or regret.
3. Draw a point outside a convex set and its projection.
4. Explain the supporting hyperplane and local inward drift.
5. Show the average update formula.
6. Expand the squared distance and identify the key inner product term.
7. Work through the regret orthant example.

This order avoids drowning the listener in quantifiers too early. The theorem is much easier to absorb once the geometric picture is vivid.

If you are giving a seminar talk, drawing the projection diagram is probably the single most useful thing you can do.

---

## 30. Historical significance

Blackwell's 1956 paper is historically important because it generalized repeated-game reasoning from scalar values to vector objectives in a way that was both elegant and constructive. It anticipated themes that later became central in control, optimization, and machine learning:

- geometric control of averaged trajectories,
- adversarial robustness,
- decomposition of complex objectives into vector constraints,
- conversion of repeated-game guarantees into learning algorithms.

Many classical results are remembered for a formula or a bound. Blackwell's theorem is remembered for a way of thinking. It teaches us to replace "optimize a number" by "steer a vector into a desirable set," and then to use geometry to determine whether that steering is possible.

---

## 31. A fuller worked regret derivation

Because regret minimization is such an important application, it is worth spelling out one derivation more carefully.

Let there be $K$ actions. At round $t$, the learner chooses a mixed action $x_t \in \Delta(K)$, then the environment reveals a loss vector $\ell_t \in [0,1]^K$, and the learner samples an action $a_t \sim x_t$. Define the stage vector payoff by

$$
g(x_t, \ell_t) = \big( \langle x_t, \ell_t \rangle - \ell_t(1), \dots, \langle x_t, \ell_t \rangle - \ell_t(K) \big).
$$

This is the vector of instantaneous regrets relative to each fixed action.

The target is the negative orthant $\mathbb{R}_{-}^K$. Suppose the current average regret vector is $r \notin \mathbb{R}_{-}^K$. Its projection onto the orthant is the vector $r^- = (\min\{r_1,0\},\dots,\min\{r_K,0\})$. Thus the outward normal is the positive part $r^+ = r-r^-$. Blackwell's halfspace condition asks for a mixed action $x$ satisfying, for every loss vector $\ell$,

$$
\langle r^+, g(x,\ell) - r^- \rangle \le 0.
$$

Since coordinates where $r_k \le 0$ contribute nothing to $r^+$, and $r^-_k = 0$ whenever $r_k^+ > 0$, this reduces to

$$
\sum_{k=1}^K r_k^+ \big( \langle x,\ell \rangle - \ell(k) \big) \le 0.
$$

Rearranging,

$$
\left(\sum_{k=1}^K r_k^+\right) \langle x,\ell \rangle
\le \sum_{k=1}^K r_k^+ \ell(k).
$$

This is satisfied by choosing

$$
x(k) = \frac{r_k^+}{\sum_m r_m^+}
$$

when the denominator is positive. Indeed, then the left-hand side equals the right-hand side identically.

Thus the Blackwell strategy for approaching the negative orthant is exactly regret matching.

This derivation is satisfying because it makes the update rule look inevitable rather than mysterious. The weights on actions are simply the positive coordinates of current regret, normalized into a distribution.

---

## 32. Why approachability is still worth learning today

A student might ask: if modern online learning already has mirror descent, optimism, adaptive regularization, and so on, why should I still learn Blackwell approachability?

There are several good reasons.

### 32.1 It provides structural insight

Approachability tells you what is really going on geometrically in many sequential decision problems. Instead of memorizing a particular algorithm, you understand the shape of the guarantee.

### 32.2 It is highly reusable

When confronted with a new problem involving multiple constraints, fairness conditions, risk metrics, or comparison classes, approachability often gives a general recipe. You ask: can I encode the desired properties as a target set for a vector payoff?

### 32.3 It unifies fields

The theorem connects game theory, online learning, control, convex analysis, and optimization. Few results are so compact yet so interdisciplinary.

### 32.4 It teaches good mathematical habits

Blackwell's theorem is an excellent example of finding the right invariant and the right geometric representation. Learning it sharpens intuition for projection arguments, minimax logic, and Lyapunov-style analysis.

In short, even if you never implement the canonical Blackwell strategy directly, the theorem can shape how you formulate and solve problems.

---

## 33. Suggested mental model

If you want one mental model to remember after reading these notes, use this one:

1. The repeated game generates a cloud of possible vector payoffs.
2. Averaging turns the trajectory into a slowly moving point in space.
3. The target set encodes the long-run property you want.
4. The projection identifies how the current average violates that property.
5. The supporting hyperplane turns that violation into a scalar direction.
6. A one-shot minimax response neutralizes drift in that direction.
7. Repeating this forever makes the average converge toward the set.

That is Blackwell approachability in a nutshell.

---

## 34. Concluding summary

Blackwell approachability is the theory of steering average vector payoffs toward a desired set in an adversarial repeated game. Its classical convex-set theorem says that a closed convex target set is approachable exactly when, informally speaking, player 1 can always respond so that the expected next payoff lies on the correct side of the supporting hyperplane at the current projection point.

The theorem matters because it turns a long-run strategic control problem into a geometric one-shot criterion. The proof is built from three ingredients:

- projection onto a convex set,
- a supporting hyperplane that identifies the offending direction,
- a distance-squared recursion showing nonpositive drift plus vanishing noise.

Approachability generalizes scalar minimax theory, explains no-regret learning and regret matching, connects to calibration and equilibrium computation, and continues to influence modern online optimization.

If you understand the following sentence, you understand the core of the subject:

> When the average payoff is outside the target set, project it back to the set, look at the outward normal, and play so the next expected payoff does not move farther outward in that direction.

Everything else is refinement, application, or generalization.

---

## 35. Further reading

For a first serious pass beyond these notes, a good reading list is:

1. **D. Blackwell (1956),** *An analog of the minimax theorem for vector payoffs.* The original source.
2. Surveys or lecture notes on **approachability in repeated games** for modern formulations and proofs.
3. **Hart and Mas-Colell** on regret matching and correlated equilibrium.
4. Texts or notes on **online learning and regret minimization**, where approachability appears as a foundational perspective.
5. Work on **calibration** and **approachability under partial monitoring** if you want to see how the theory extends when feedback is limited.

A productive way to study the topic further is to alternate between theorem statements and concrete applications. The abstract geometry becomes much clearer once you repeatedly see it instantiate as regret minimization, calibration, and constrained learning.

---

## 36. Short checklist of key ideas

To close, here is a compact checklist of the main ideas you should retain.

- A vector payoff repeated game produces average outcomes in $\mathbb{R}^d$.
- The goal is to approach a target set $S$, not optimize a scalar.
- Closed convex sets are the natural targets.
- Projection onto $S$ identifies the current error.
- The outward normal at the projection defines a local scalarized game.
- If player 1 can make the expected next payoff lie in the supporting halfspace, then distance to $S$ decreases in expectation.
- This condition is both sufficient and, for convex sets, necessary.
- External regret, internal regret, and calibration can all be cast as approachability problems.
- The negative orthant is the canonical target set for regret minimization.
- Blackwell's theorem is a geometric ancestor of modern online learning.

If those points feel natural, then you have a solid entry-level graduate understanding of Blackwell approachability.
