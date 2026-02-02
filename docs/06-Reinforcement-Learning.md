# Chapter 6: Reinforcement Learning

> *"The key to artificial intelligence has always been the representation."*
> — Jeff Hawkins

Reinforcement Learning (RL) is the science of decision-making under uncertainty. Unlike supervised learning where we learn from labeled examples, RL agents learn by interacting with an environment, receiving feedback through rewards, and discovering optimal behaviors through trial and error. This chapter provides a comprehensive treatment of RL, starting with the foundational Markov Decision Process framework.

---

## Markov Decision Processes

A **Markov Decision Process (MDP)** is the mathematical framework that underlies reinforcement learning. It provides a formal model for sequential decision-making where outcomes are partly random and partly controlled by an agent. MDPs are widely used in **reinforcement learning, robotics, finance, and game theory**.

### Components of an MDP

An MDP consists of the following components:

- **States ($S$):** A set of all possible states the agent can be in.
- **Actions ($A$):** A set of all possible actions the agent can take.
- **Transition Probability ($P$):** The probability of moving from one state to another given an action.
- **Reward Function ($R$):** The reward received for taking an action in a particular state.
- **Policy ($\pi$):** A strategy that defines the action selection process in each state.
- **Discount Factor ($\gamma$):** A factor that determines how much future rewards are valued compared to immediate rewards.

Mathematically, an MDP is represented as a tuple:

$$
(S, A, P, R, \pi, \gamma)
$$

where:

- $S$ is the set of states.
- $A$ is the set of actions.
- $P(s' | s, a)$ is the probability of transitioning from state $s$ to state $s'$ when taking action $a$.
- $R(s, a)$ is the reward received after taking action $a$ in state $s$.
- $\pi(a | s)$ is the policy, which defines the probability of taking action $a$ in state $s$.
- $\gamma \in [0,1]$ is the discount factor.

### State Transition Probability

The probability of moving from state $s$ to state $s'$ after taking action $a$ is given by:

$$
P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)
$$

This represents the **Markov Property**, meaning the next state **only depends on the current state and action**, not the previous history.

### Policy in MDP

A **policy ($\pi$)** defines the agent's behavior by mapping states to actions. Policies can be:

- **Deterministic Policy ($\pi: S \to A$)**: The agent always takes a specific action in a given state.

  $$a = \pi(s)$$

- **Stochastic Policy ($\pi: S \times A \to [0,1]$)**: The agent chooses actions probabilistically.

  $$\pi(a | s) = \Pr(A_t = a | S_t = s)$$

### Optimal Policy

An optimal policy $\pi^*$ maximizes the expected return:

$$
\pi^* = \arg\max_\pi V^\pi(s), \forall s \in S
$$

where $V^\pi(s)$ is the value function under policy $\pi$. The optimal policy leads to the highest cumulative rewards over time.

---

## The Bellman Equation

The **Bellman Equation** provides a recursive formula for **value functions**, helping determine the **optimal policy** that maximizes expected cumulative rewards.

### State-Value Function

The **value function** of a state $s$ under a policy $\pi$ is:

$$
V^\pi(s) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \mid S_0 = s, \pi \right]
$$

Using the **Bellman equation**, we express the recursive relationship:

$$
V^\pi(s) = \sum_{a \in A} \pi(a | s) \sum_{s' \in S} P(s' | s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]
$$

### Action-Value Function (Q-Function)

The **Q-value function** measures the expected reward for taking an action $a$ in state $s$ and then following policy $\pi$:

$$
Q^\pi(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \mid S_0 = s, A_0 = a, \pi \right]
$$

The **Bellman equation for Q-values** is:

$$
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) \sum_{a' \in A} \pi(a' | s') Q^\pi(s', a')
$$

### Bellman Optimality Equations

The goal of reinforcement learning is to find the **optimal policy** that maximizes the expected cumulative reward. The **Bellman optimality equation** for the optimal state-value function is:

$$
V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s' | s, a) \left[ R(s, a) + \gamma V^*(s') \right]
$$

Similarly, the **Bellman optimality equation for Q-values** is:

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) \max_{a' \in A} Q^*(s', a')
$$

---

## Reward-Based Learning

Reward-Based Learning is a fundamental concept in reinforcement learning where an agent learns to make decisions by receiving rewards for its actions. The agent's objective is to maximize the total cumulative reward over time.

### Types of Rewards

- **Immediate Rewards ($R(s, a)$):** A reward received immediately after taking action $a$ in state $s$.
- **Delayed Rewards:** A reward that is given after multiple steps, requiring the agent to learn long-term strategies.
- **Sparse Rewards:** Rewards that occur infrequently, making learning more challenging.
- **Dense Rewards:** Frequent rewards that provide continuous feedback.

### The Return

To maximize cumulative rewards, agents use **return ($G_t$)**, which is the sum of discounted future rewards:

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$

### Exploration vs. Exploitation

A key challenge in reward-based learning is balancing:

- **Exploration:** Trying new actions to discover potentially better rewards.
- **Exploitation:** Choosing known good actions to maximize immediate rewards.

**Methods for Managing the Trade-off:**

- **Epsilon-Greedy Strategy:** Randomly selects an action with probability $\epsilon$, otherwise chooses the best-known action.
- **Upper Confidence Bound (UCB):** Prioritizes actions with uncertain rewards.
- **Thompson Sampling:** Uses probability distributions to select actions.
- **Boltzmann Exploration:** Selects actions based on softmax over Q-values.

---

## Taxonomy of RL Algorithms

### Model-Based vs Model-Free

- **Model-Based RL**: Learns a model of environment dynamics $P(s'|s,a)$ and uses it for planning
- **Model-Free RL**: Learns directly from experience without modeling the environment

### Value-Based vs Policy-Based

- **Value-Based**: Learn value functions, derive policy implicitly (Q-learning, DQN)
- **Policy-Based**: Directly optimize the policy (REINFORCE, PPO)
- **Actor-Critic**: Combine both approaches (A2C, A3C, SAC)

### On-Policy vs Off-Policy

- **On-Policy**: Learn from data collected by current policy (SARSA, PPO)
- **Off-Policy**: Learn from data collected by any policy (Q-learning, DQN, SAC)

---

## Model-Based Reinforcement Learning

Model-Based RL refers to methods where an agent **learns a model** of the environment and uses it for decision-making. This typically involves:

1. **Learning the Transition Model ($P(s' | s, a)$)**: Predicting the next state given the current state and action.
2. **Learning the Reward Model ($R(s, a)$)**: Estimating the reward for taking an action in a given state.
3. **Planning**: Using the learned model to simulate trajectories and improve decision-making.

### Dynamic Programming

Uses Bellman equations to iteratively compute value functions, assuming a known model.

**Value Iteration:**
$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a) + \gamma V_k(s')]
$$

**Policy Iteration:**
1. Policy Evaluation: Compute $V^\pi$
2. Policy Improvement: $\pi'(s) = \arg\max_a Q^\pi(s,a)$

### Model Predictive Control (MPC)

Uses a learned model to simulate future states and optimize actions:

$$
a_{0:H}^* = \arg\max_{a_{0:H}} \sum_{t=0}^{H} \gamma^t \hat{r}(s_t, a_t)
$$

At each step:
1. Optimize action sequence using learned model
2. Execute first action
3. Re-plan at next step

### Dyna-Q

Combines model-free and model-based learning by integrating planning and learning:

1. Take action, observe transition
2. Update Q-values from real experience
3. Update model from real experience
4. Plan: Sample from model, update Q-values

### Monte Carlo Tree Search (MCTS)

MCTS is a search-based decision-making algorithm used in **game AI** and **planning problems**. It simulates future states by exploring the decision tree and backing up estimated rewards.

**MCTS Steps:**

1. **Selection**: Start from the root and select the most promising node based on UCB1:
   $$
   \text{UCT}(v) = \frac{w(v)}{n(v)} + c \sqrt{\frac{\ln N}{n(v)}}
   $$

2. **Expansion**: Add a new node to the tree when a promising unexplored action is found.

3. **Simulation (Rollout)**: Play out random simulations to estimate the value of the new state.

4. **Backpropagation**: Update the values of visited nodes based on the simulation outcome.

**Applications of MCTS:**
- AlphaGo (DeepMind)
- Chess and board games
- Robotics planning and decision-making

### World Models (Dreamer)

State-of-the-art model-based RL:
1. **World Model**: Recurrent State-Space Model (RSSM)
2. **Imagination**: Roll out trajectories in latent space
3. **Actor-Critic**: Train on imagined trajectories

---

## Model-Free Reinforcement Learning

Model-Free RL does not explicitly learn a model of the environment. Instead, the agent learns directly from experience by interacting with the environment and optimizing rewards.

### Value-Based Methods

#### Q-Learning

Q-learning is a foundational off-policy algorithm that learns the optimal action-value function:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

**Key Properties:**
- Off-policy: Uses max over next actions (greedy)
- Tabular: Works with discrete state-action spaces
- Convergence: Guaranteed under certain conditions

#### SARSA (State-Action-Reward-State-Action)

SARSA is an on-policy variant that uses the actual next action:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

**Difference from Q-learning:** Uses $Q(s', a')$ instead of $\max_{a'} Q(s', a')$, making it more conservative.

#### Deep Q-Networks (DQN)

DQN revolutionized RL by using deep neural networks to approximate Q-values:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

**Key Innovations:**
1. **Experience Replay**: Store transitions in buffer, sample randomly to break correlations
2. **Target Network**: Separate network $\theta^-$ updated periodically for stability
3. **Frame Stacking**: Stack consecutive frames to capture motion

#### Double DQN

Addresses overestimation bias in DQN by decoupling action selection and evaluation:

$$
Y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

#### Dueling DQN

Separates Q-value into state value and advantage:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|A|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
$$

#### Prioritized Experience Replay

Samples important transitions more frequently based on TD-error:

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon
$$

#### Rainbow DQN

Combines multiple improvements:
- Double Q-learning
- Prioritized replay
- Dueling networks
- Multi-step learning
- Distributional RL
- Noisy networks

---

## Policy Gradient Methods

### The Policy Gradient Theorem

The gradient of expected return with respect to policy parameters:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \right]
$$

### REINFORCE

The simplest policy gradient algorithm using Monte Carlo returns:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]
$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return from time $t$.

**Variance Reduction with Baseline:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) (Q^{\pi}(s, a) - b(s)) \right]
$$

### Advantage Actor-Critic (A2C)

Uses advantage function to reduce variance:

$$
A(s, a) = Q(s, a) - V(s)
$$

**Actor Update:** $\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a|s) A(s, a)$

**Critic Update:** $\phi \leftarrow \phi - \alpha_\phi \nabla_\phi (V_\phi(s) - G_t)^2$

### Asynchronous Advantage Actor-Critic (A3C)

Parallelizes A2C across multiple workers:
- Each worker interacts with its own environment copy
- Asynchronous gradient updates to shared parameters
- Diverse experience through different exploration

### Proximal Policy Optimization (PPO)

PPO constrains policy updates to prevent destructive large steps:

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

**Why PPO is Popular:**
- Simple to implement
- Good sample efficiency
- Stable training
- Works across many domains

### Trust Region Policy Optimization (TRPO)

Constrains policy updates using KL divergence:

$$
\max_\theta \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t \right] \quad \text{s.t.} \quad \mathbb{E}_t[D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta
$$

---

## Generalized Policy Optimization (GPO) Methods

GPO methods represent a family of algorithms that generalize and extend policy optimization techniques, particularly for training large language models with human feedback.

### Direct Preference Optimization (DPO)

DPO eliminates the need for a separate reward model by directly optimizing the policy from preference data:

$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

where:
- $y_w$ = preferred (winning) response
- $y_l$ = dispreferred (losing) response
- $\pi_{ref}$ = reference policy (typically SFT model)
- $\beta$ = temperature parameter

**Advantages over RLHF:**
- No reward model training needed
- No RL optimization loop
- Stable and efficient training
- Mathematically equivalent to RLHF under Bradley-Terry model

### Group Relative Policy Optimization (GRPO)

GRPO improves upon PPO for LLM alignment by using group-based advantage estimation:

$$
\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|x)} \left[ \sum_{i=1}^G \min \left( r_i(\theta) \hat{A}_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]
$$

**Group-based Advantage:**
$$
\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)}
$$

### Identity Preference Optimization (IPO)

IPO addresses the overfitting issues in DPO:

$$
\mathcal{L}_{IPO}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]
$$

### Kahneman-Tversky Optimization (KTO)

KTO uses prospect theory to model human preferences:

$$
\mathcal{L}_{KTO}(\theta) = \mathbb{E}_{x, y} \left[ w(y) \cdot \left( 1 - v_\theta(x, y) \right) \right]
$$

**Key Features:**
- Works with unpaired preference data
- Models loss aversion ($\lambda > 1$)
- More data-efficient

### Odds Ratio Preference Optimization (ORPO)

ORPO combines SFT and preference optimization in a single stage:

$$
\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}
$$

**Advantages:**
- Single-stage training (no separate SFT then alignment)
- No reference model needed
- Memory efficient

### Reinforcement Learning from Human Feedback (RLHF)

The foundational approach that GPO methods improve upon:

**Three-Stage Pipeline:**

1. **Supervised Fine-Tuning (SFT):**
$$
\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{demo}} [\log \pi_\theta(y|x)]
$$

2. **Reward Model Training:**
$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]
$$

3. **RL Fine-Tuning (typically PPO):**
$$
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} [r_\phi(x, y)] - \beta D_{KL}(\pi_\theta || \pi_{ref})
$$

### Comparison of GPO Methods

| Method | Requires RM | Paired Data | Reference Model | Stages |
|:-------|:------------|:------------|:----------------|:-------|
| **RLHF (PPO)** | Yes | Yes | Yes | 3 |
| **DPO** | No | Yes | Yes | 2 |
| **GRPO** | Yes | No | Yes | 2 |
| **IPO** | No | Yes | Yes | 2 |
| **KTO** | No | No | Yes | 2 |
| **ORPO** | No | Yes | No | 1 |

---

## Actor-Critic Methods

### Soft Actor-Critic (SAC)

SAC maximizes both expected return and entropy for exploration:

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]
$$

**Key Components:**
- **Entropy Regularization**: Encourages exploration
- **Twin Q-Networks**: Two critics to reduce overestimation
- **Automatic Temperature Tuning**: Learns optimal $\alpha$

### Twin Delayed DDPG (TD3)

Addresses overestimation in DDPG with three key tricks:

1. **Clipped Double Q-Learning**: Use minimum of two critics
$$
y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \tilde{a}')
$$

2. **Delayed Policy Updates**: Update actor less frequently than critics

3. **Target Policy Smoothing**: Add noise to target actions
$$
\tilde{a}' = \pi_{\phi'}(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

### Deep Deterministic Policy Gradient (DDPG)

Extends DQN to continuous action spaces:

- **Actor**: Deterministic policy $\mu_\theta(s)$
- **Critic**: Q-function $Q_\phi(s, a)$

Actor Update:
$$
\nabla_\theta J \approx \mathbb{E}_s \left[ \nabla_a Q_\phi(s, a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]
$$

---

## Exploration Strategies

### Epsilon-Greedy

$$
a = \begin{cases} \arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon \\ \text{random action} & \text{with probability } \epsilon \end{cases}
$$

### Upper Confidence Bound (UCB)

$$
a = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln t}{N(s, a)}} \right]
$$

### Intrinsic Motivation

Add curiosity-driven rewards:
- **Prediction Error**: Reward for surprising states
- **Information Gain**: Reward for reducing uncertainty
- **State Visitation Count**: Reward for novel states

### Noisy Networks

Replace linear layers with noisy versions for parameter-space exploration.

---

## Multi-Agent Reinforcement Learning

### Independent Learners

Each agent learns independently, treating others as part of environment.

### Centralized Training, Decentralized Execution (CTDE)

- Train with global information
- Execute with local observations only

### QMIX

Factorizes joint Q-function:
$$
Q_{tot}(\boldsymbol{\tau}, \boldsymbol{a}) = f_\theta(Q_1(\tau_1, a_1), ..., Q_n(\tau_n, a_n))
$$

---

## Offline Reinforcement Learning

Learn from fixed datasets without environment interaction.

### Conservative Q-Learning (CQL)

Adds penalty for out-of-distribution actions:
$$
\mathcal{L}_{CQL} = \alpha \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta} [Q(s, a)] \right] + \mathcal{L}_{TD}
$$

### Decision Transformer

Frames RL as sequence modeling:
- Input: (Return-to-go, State, Action) sequences
- Output: Next action
- Architecture: GPT-style transformer

---

## Applications of Reinforcement Learning

| Domain | Application | Notable Examples |
|:-------|:------------|:-----------------|
| **Games** | Game playing | AlphaGo, AlphaStar, OpenAI Five |
| **Robotics** | Manipulation, locomotion | Boston Dynamics, manipulation tasks |
| **Autonomous Vehicles** | Navigation, decision-making | Waymo, Tesla Autopilot |
| **Finance** | Trading, portfolio optimization | Algorithmic trading systems |
| **Healthcare** | Treatment optimization | Drug dosing, clinical trials |
| **NLP** | RLHF for language models | ChatGPT, Claude |
| **Recommendation** | Personalization | Netflix, YouTube |

## Summary: Algorithm Selection Guide

| Algorithm | Action Space | Sample Efficiency | Stability | Use Case |
|:----------|:-------------|:------------------|:----------|:---------|
| **DQN** | Discrete | Medium | Medium | Atari games |
| **PPO** | Both | Medium | High | General purpose |
| **SAC** | Continuous | High | High | Robotics |
| **TD3** | Continuous | High | High | Robotics |
| **Dreamer** | Both | Very High | Medium | Complex environments |
| **Decision Transformer** | Both | N/A (Offline) | High | Offline RL |

---

*For further reading: Sutton & Barto's "Reinforcement Learning: An Introduction" and OpenAI Spinning Up documentation.*
