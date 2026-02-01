# Chapter 5: Reinforcement Learning - Theory and Applications

> *"The key to artificial intelligence has always been the representation."*
> — Jeff Hawkins

Reinforcement Learning (RL) is the science of decision-making under uncertainty. Unlike supervised learning where we learn from labeled examples, RL agents learn by interacting with an environment, receiving feedback through rewards, and discovering optimal behaviors through trial and error.

## Fundamentals of Reinforcement Learning

### Key Components

The RL framework consists of several fundamental components:

| Component | Symbol | Description |
|:----------|:-------|:------------|
| **Agent** | - | The decision-maker that interacts with the environment |
| **Environment** | - | The external system the agent operates in |
| **State** | $s \in S$ | A representation of the environment at time $t$ |
| **Action** | $a \in A$ | A decision taken by the agent |
| **Reward** | $r \in \mathbb{R}$ | Scalar feedback signal |
| **Policy** | $\pi(a|s)$ | Probability of taking action $a$ in state $s$ |
| **Value Function** | $V^\pi(s)$ | Expected cumulative reward from state $s$ |
| **Q-Function** | $Q^\pi(s,a)$ | Expected cumulative reward for action $a$ in state $s$ |
| **Discount Factor** | $\gamma \in [0,1]$ | Importance of future rewards |

### The RL Objective

The goal is to find an optimal policy $\pi^*$ that maximizes expected cumulative discounted reward:

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

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

## Value-Based Methods

### Q-Learning

Q-learning is a foundational off-policy algorithm that learns the optimal action-value function:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

**Key Properties:**
- Off-policy: Uses max over next actions (greedy)
- Tabular: Works with discrete state-action spaces
- Convergence: Guaranteed under certain conditions

### SARSA (State-Action-Reward-State-Action)

SARSA is an on-policy variant that uses the actual next action:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

**Difference from Q-learning:** Uses $Q(s', a')$ instead of $\max_{a'} Q(s', a')$, making it more conservative.

### Deep Q-Networks (DQN)

DQN revolutionized RL by using deep neural networks to approximate Q-values:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

**Key Innovations:**
1. **Experience Replay**: Store transitions in buffer, sample randomly to break correlations
2. **Target Network**: Separate network $\theta^-$ updated periodically for stability
3. **Frame Stacking**: Stack consecutive frames to capture motion

### Double DQN

Addresses overestimation bias in DQN by decoupling action selection and evaluation:

$$
Y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
$$

### Dueling DQN

Separates Q-value into state value and advantage:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|A|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
$$

This architecture learns which states are valuable without needing to learn the effect of each action.

### Prioritized Experience Replay

Samples important transitions more frequently based on TD-error:

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon
$$

where $\delta_i$ is the TD-error for transition $i$.

### Rainbow DQN

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

Uses conjugate gradient and line search for optimization.

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

**Updates:**

Critic: $\mathcal{L}_Q = \mathbb{E} \left[ \left( Q(s,a) - (r + \gamma (Q_{target}(s', a') - \alpha \log \pi(a'|s'))) \right)^2 \right]$

Actor: $\mathcal{L}_\pi = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ \alpha \log \pi(a|s) - Q(s, a) \right]$

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

## Model-Based Reinforcement Learning

### World Models

Learn a model of the environment and use it for planning:

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

**Components:**
1. **Dynamics Model**: Predicts next state
2. **Reward Model**: Predicts reward
3. **Planning**: Use model for lookahead (MPC, MCTS)

### Dyna-Q

Integrates learning and planning:
1. Take action, observe transition
2. Update Q-values from real experience
3. Update model from real experience
4. Plan: Sample from model, update Q-values

### Model Predictive Control (MPC)

At each step:
1. Optimize action sequence using learned model
2. Execute first action
3. Re-plan at next step

$$
a_{0:H}^* = \arg\max_{a_{0:H}} \sum_{t=0}^{H} \gamma^t \hat{r}(s_t, a_t)
$$

### Dreamer

State-of-the-art model-based RL:
1. **World Model**: Recurrent State-Space Model (RSSM)
2. **Imagination**: Roll out trajectories in latent space
3. **Actor-Critic**: Train on imagined trajectories

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

Replace linear layers with noisy versions:
$$
y = (W + \sigma_{W} \odot \epsilon_W) x + (b + \sigma_b \odot \epsilon_b)
$$

---

## Multi-Agent Reinforcement Learning (MARL)

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

where $f_\theta$ is a monotonic mixing network.

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
