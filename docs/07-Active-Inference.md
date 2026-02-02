# Chapter 7: Active Inference

> *"The brain is a prediction machine."*
> — Karl Friston

Active Inference is a unified theory of brain function that extends the Free Energy Principle to action and decision-making. Unlike traditional reinforcement learning which separates perception from action, Active Inference treats both as aspects of the same underlying process: minimizing surprise (or equivalently, maximizing model evidence).

---

## The Free Energy Principle

The Free Energy Principle (FEP) proposes that all living systems minimize their **variational free energy**—a quantity that bounds surprise. This principle provides a normative framework for understanding perception, learning, and action.

### Surprise and Self-Organization

Living systems maintain their organization by occupying a limited set of states (their phenotype). From an information-theoretic perspective, this means minimizing **surprise**:

$$
\text{Surprise} = -\log p(o | m)
$$

where $o$ represents observations and $m$ is the agent's generative model.

### Variational Free Energy

Since surprise is often intractable to compute directly, we use **variational free energy** as an upper bound:

$$
F = \underbrace{D_{KL}[q(\theta) || p(\theta | o)]}_{\text{Divergence}} + \underbrace{(-\log p(o | m))}_{\text{Surprise}}
$$

Or equivalently:

$$
F = \underbrace{\mathbb{E}_q[\log q(\theta) - \log p(\theta)]}_{\text{Complexity}} - \underbrace{\mathbb{E}_q[\log p(o | \theta)]}_{\text{Accuracy}}
$$

This formulation reveals that minimizing free energy involves balancing **accuracy** (explaining observations) with **complexity** (staying close to prior beliefs).

---

## Generative Models in Active Inference

Active Inference agents maintain a **generative model** of how their observations are generated. This model captures both:

1. **Hidden states** of the world
2. **How actions influence future states**

### Partially Observable Markov Decision Process (POMDP)

The standard formulation uses a POMDP structure:

- **Hidden States**: $s_t$ (what the agent believes about the world)
- **Observations**: $o_t$ (sensory input)
- **Actions**: $a_t$ (motor output)
- **Policies**: $\pi$ (sequences of actions)

The generative model factorizes as:

$$
p(o_{1:T}, s_{1:T}, \pi) = p(\pi) \prod_{t=1}^{T} p(o_t | s_t) p(s_t | s_{t-1}, a_{t-1})
$$

### Belief Updating

Perception corresponds to **state estimation**—inferring hidden states from observations:

$$
q(s_t) = \sigma(\log p(o_t | s_t) + \log p(s_t | s_{t-1}, a_{t-1}))
$$

where $\sigma$ is the softmax function ensuring beliefs form a proper distribution.

---

## Expected Free Energy

The key innovation of Active Inference is using **Expected Free Energy (EFE)** to evaluate and select policies. EFE quantifies how much free energy a policy is expected to produce in the future:

$$
G(\pi) = \sum_{\tau} \mathbb{E}_{q(o_\tau, s_\tau | \pi)} [\log q(s_\tau | \pi) - \log p(o_\tau, s_\tau | \pi)]
$$

### Decomposition of EFE

EFE naturally decomposes into epistemic and instrumental components:

$$
G(\pi) = \underbrace{-\mathbb{E}_q[D_{KL}[q(s_\tau | o_\tau, \pi) || q(s_\tau | \pi)]]}_{\text{Epistemic Value (Information Gain)}} + \underbrace{\mathbb{E}_q[\log q(o_\tau) - \log p(o_\tau)]}_{\text{Pragmatic Value (Goal-Seeking)}}
$$

**Epistemic Value**: Measures expected information gain—policies that reduce uncertainty about hidden states have low (negative) epistemic value.

**Pragmatic Value**: Measures alignment with preferred outcomes—policies leading to desired observations have low pragmatic value.

### Alternative Decomposition

$$
G(\pi) = \underbrace{\mathbb{E}_q[H[p(o_\tau | s_\tau)]]}_{\text{Ambiguity}} + \underbrace{D_{KL}[q(o_\tau | \pi) || p(o_\tau)]}_{\text{Risk}}
$$

**Ambiguity**: Expected uncertainty about observations given states.

**Risk**: Divergence between expected and preferred observations.

---

## Policy Selection

Policies are selected by treating them as hidden variables and inferring the most likely policy given the generative model:

$$
p(\pi) = \sigma(-\gamma \cdot G(\pi))
$$

where $\gamma$ is a precision parameter (inverse temperature) controlling the exploration-exploitation trade-off.

### Action Selection

Given the posterior over policies, actions are selected by marginalizing:

$$
p(a_t) = \sum_{\pi} p(a_t | \pi) p(\pi)
$$

This naturally balances:
- **Exploitation**: Selecting actions that achieve goals
- **Exploration**: Selecting actions that resolve uncertainty

---

## Comparison with Reinforcement Learning

Active Inference and RL share many features but differ fundamentally in their approach:

| Aspect | Reinforcement Learning | Active Inference |
|:-------|:----------------------|:-----------------|
| **Objective** | Maximize cumulative reward | Minimize expected free energy |
| **Exploration** | Separate mechanism (ε-greedy, UCB) | Emergent from epistemic value |
| **Value Function** | Learned from rewards | Derived from generative model |
| **Preferences** | Encoded as reward function | Encoded as prior over observations |
| **Model** | Optional (model-free methods exist) | Central to the framework |
| **Uncertainty** | Often ignored or handled separately | Explicitly represented and used |

### Reward as Prior Preference

In Active Inference, rewards are reconceptualized as **prior preferences** over observations:

$$
\log p(o) \propto r(o)
$$

This means goal-directed behavior emerges from minimizing the divergence between expected and preferred outcomes, rather than maximizing an external reward signal.

### Information-Seeking Behavior

One key advantage of Active Inference is that curiosity and exploration emerge naturally. The epistemic component of EFE drives the agent to seek information that reduces uncertainty about the world.

---

## Mathematical Formulation

### State-Space Model

The generative model can be expressed as:

**Likelihood (Observation Model):**
$$
p(o_t | s_t) = \text{Cat}(\mathbf{A} \cdot s_t)
$$

**Transition Model:**
$$
p(s_{t+1} | s_t, a_t) = \text{Cat}(\mathbf{B}_{a_t} \cdot s_t)
$$

**Prior over Initial States:**
$$
p(s_1) = \text{Cat}(\mathbf{D})
$$

**Prior over Policies:**
$$
p(\pi) = \sigma(-\gamma \cdot G(\pi)) \cdot p_0(\pi)
$$

### Message Passing

Inference in Active Inference can be implemented via **message passing** on factor graphs:

$$
\mu_{s_t} \propto \mathbf{A}^T o_t \odot \mathbf{B}_{a_{t-1}} \mu_{s_{t-1}} \odot \mathbf{B}_{a_t}^T \mu_{s_{t+1}}
$$

where $\odot$ denotes element-wise multiplication.

---

## Deep Active Inference

Recent work has extended Active Inference to work with deep neural networks for handling complex, high-dimensional observations.

### Variational Autoencoders for Perception

Use VAEs to learn the observation model:

$$
q_\phi(s | o) \approx p(s | o)
$$

The encoder maps observations to beliefs about hidden states.

### Deep Transition Models

Learn transition dynamics using neural networks:

$$
p_\theta(s_{t+1} | s_t, a_t)
$$

This enables scaling to complex environments.

### Amortized Policy Inference

Instead of computing EFE for all policies, learn a policy network:

$$
\pi_\psi(a | s) \approx \arg\min_\pi G(\pi | s)
$$

This enables tractable policy selection in continuous action spaces.

---

## Hierarchical Active Inference

Active Inference naturally extends to hierarchical models where higher levels encode slower temporal dynamics and more abstract representations.

### Temporal Hierarchy

$$
p(o, s^{(1)}, s^{(2)}, ..., s^{(L)}) = p(o | s^{(1)}) \prod_{l=1}^{L-1} p(s^{(l)} | s^{(l+1)}) p(s^{(L)})
$$

Higher levels:
- Operate on slower timescales
- Encode more abstract goals
- Provide context for lower levels

### Goal Setting

Higher levels can set goals for lower levels by specifying preferred observations:

$$
p(o^{(l)}) \leftarrow \text{message from level } l+1
$$

This enables hierarchical planning and multi-scale decision making.

---

## Applications

### Robotics

Active Inference provides a principled framework for robot control that:
- Integrates perception and action
- Handles uncertainty naturally
- Enables curiosity-driven exploration

### Neuroscience

Active Inference offers a unified account of:
- Perception as inference
- Attention as precision optimization
- Action as prediction error minimization

### Cognitive Science

The framework explains:
- Curiosity and play
- Intrinsic motivation
- Goal-directed behavior

### Artificial Intelligence

Potential applications include:
- Embodied AI agents
- Sample-efficient learning
- Safe exploration in unknown environments

---

## Relationship to World Models

Active Inference is closely related to world model approaches in deep RL:

| World Models | Active Inference |
|:-------------|:-----------------|
| Learn dynamics model | Learn generative model |
| Plan in latent space | Evaluate policies via EFE |
| Maximize reward | Minimize free energy |
| Separate exploration mechanism | Unified exploration-exploitation |

Both approaches emphasize:
- **Model-based reasoning**: Using learned models for planning
- **Latent representations**: Working in compressed state spaces
- **Prediction**: Anticipating future observations

The key difference is that Active Inference derives both perception and action from a single principle (free energy minimization), while world model approaches typically treat learning, planning, and exploration as separate problems.

---

## Key Equations Summary

| Concept | Equation |
|:--------|:---------|
| **Free Energy** | $F = D_{KL}[q(\theta) \| p(\theta \| o)] - \log p(o)$ |
| **Expected Free Energy** | $G(\pi) = \mathbb{E}_q[\log q(s_\tau \| \pi) - \log p(o_\tau, s_\tau \| \pi)]$ |
| **Policy Selection** | $p(\pi) = \sigma(-\gamma \cdot G(\pi))$ |
| **Epistemic Value** | $-D_{KL}[q(s \| o) \| q(s)]$ |
| **Pragmatic Value** | $D_{KL}[q(o) \| p(o)]$ |

---

## Future Directions

Active Inference is an active area of research with several exciting directions:

1. **Scalability**: Extending to larger state and action spaces
2. **Multi-agent systems**: Active Inference for social cognition
3. **Continual learning**: Updating generative models over lifetimes
4. **Hybrid approaches**: Combining with deep RL techniques
5. **Real-world robotics**: Practical implementations for physical systems

---

*For further reading: Karl Friston's work on the Free Energy Principle, and the Active Inference textbook by Parr, Pezzulo, and Friston.*
