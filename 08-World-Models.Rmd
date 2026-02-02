# Chapter 8: World Models

> *"An agent's world model allows it to imagine the future and plan actions accordingly."*
> — David Ha & Jürgen Schmidhuber

World Models represent one of the most significant paradigms in artificial intelligence—the idea that intelligent agents build internal representations of their environment that allow them to simulate, predict, and plan. This chapter traces the evolution of world models from foundational work to cutting-edge approaches like JEPA, DINO-WM, and Genie.

---

## Foundations: Ha & Schmidhuber's World Models

### The Original Vision

In their seminal 2018 paper, David Ha and Jürgen Schmidhuber introduced the **World Models** framework, demonstrating that agents could learn to act by first learning to dream. Their key insight was to decompose the problem into three components:

1. **Vision Model (V)**: Compresses high-dimensional observations into compact latent representations
2. **Memory Model (M)**: Predicts future latent states, enabling simulation
3. **Controller (C)**: A simple policy that operates on the compressed representations

### Architecture

The original architecture combined:

**Variational Autoencoder (VAE)** for perception:
$$
z = \text{Encoder}(x), \quad \hat{x} = \text{Decoder}(z)
$$

**MDN-RNN** for dynamics prediction:
$$
P(z_{t+1} | a_t, z_t, h_t) = \sum_{i=1}^{K} \pi_i \mathcal{N}(\mu_i, \sigma_i)
$$

where the mixture density network outputs parameters for a Gaussian mixture model over future latent states.

**Simple Linear Controller**:
$$
a_t = W_c [z_t, h_t] + b_c
$$

### Training in Dreams

A revolutionary aspect was **training entirely in imagination**:

1. Train V and M on real environment data
2. Generate simulated rollouts using M
3. Train C on these imagined trajectories
4. Transfer to real environment

This approach achieved state-of-the-art results on car racing and other control tasks while requiring orders of magnitude less real environment interaction.

### Key Insights

- **Compression is crucial**: Good latent representations enable efficient planning
- **Dreams can teach**: Simulated experience transfers to reality
- **Simplicity at the top**: Complex world models enable simple controllers
- **Uncertainty matters**: Modeling stochasticity prevents overfitting to the model

---

## Dreamer: Scalable World Models

### DreamerV1 (2019)

Dreamer extended the World Models framework with:

- **Recurrent State-Space Model (RSSM)**: Combines deterministic and stochastic latent states
- **Actor-Critic Learning**: Learns value functions and policies from imagined rollouts
- **Latent Imagination**: Plans entirely in latent space

The RSSM separates latent state into:
$$
s_t = [h_t, z_t]
$$

where $h_t$ is deterministic (captures history) and $z_t$ is stochastic (captures uncertainty).

### DreamerV2 (2020)

Improved upon V1 with:

- **Discrete Latents**: Uses categorical distributions instead of Gaussians
- **KL Balancing**: Better optimization of the variational bound
- **Larger Models**: Scaled to complex 3D environments

### DreamerV3 (2023)

Achieved human-level performance across diverse domains:

- **Symlog Predictions**: Handles varying reward scales
- **Percentile Scaling**: Normalizes returns adaptively
- **Universal Recipe**: Same hyperparameters across all tasks

Key results:
- First algorithm to collect diamonds in Minecraft from scratch
- Matched human performance on Atari benchmarks
- Strong results on continuous control

---

## JEPA: Joint Embedding Predictive Architecture

### The JEPA Paradigm

Proposed by Yann LeCun and colleagues, JEPA represents a shift from pixel-level prediction to **representation-level prediction**. Instead of predicting future observations directly, JEPA predicts future representations.

### Core Principle

$$
\text{Predict: } \hat{z}_{\text{target}} = f_\theta(z_{\text{context}})
$$

Rather than:
$$
\text{Reconstruct: } \hat{x}_{\text{target}} = g_\phi(z_{\text{context}})
$$

This avoids the need to model irrelevant details (textures, lighting) and focuses on semantic content.

### Architecture Components

- **Context Encoder**: Processes visible/past information to generate context representation
- **Target Encoder**: Processes target region/frame (used for training signal only)
- **Predictor**: Maps context representation to predicted target representation

The training objective:
$$
\mathcal{L} = \| f_\theta(E_c(x_{\text{context}})) - \text{sg}(E_t(x_{\text{target}})) \|^2
$$

where $\text{sg}$ denotes stop-gradient (target encoder is updated via EMA).

### I-JEPA: Image-based JEPA

I-JEPA applies JEPA to images through:

**Multi-block Masking**:
- Mask multiple non-overlapping target regions
- Use remaining visible regions as context
- Predict representations of masked regions

**Key Design Choices**:
- Large target blocks (encourage semantic prediction)
- Asymmetric architecture (predictor is lightweight)
- No data augmentation needed

**Results**:
- Competitive with contrastive methods (DINO, MAE)
- Better linear probe performance
- More semantically meaningful representations

### MC-JEPA: Motion-Content JEPA

Extends to video by learning:
- **Motion features**: Captured via optical flow prediction
- **Content features**: Captured via frame prediction

Architecture includes:
- Shared encoder for temporal features
- Flow estimator for motion dynamics
- Content learner for static scene understanding

### V-JEPA: Video JEPA

Predicts future frame representations from past frames:

$$
\hat{z}_{t+k} = \text{Predictor}(z_{t-n:t})
$$

**Key Features**:
- Temporal masking strategies
- Learns both short-term and long-term dependencies
- No need for labeled data or augmentations

**Applications**:
- Action recognition
- Video understanding
- Robotic planning

---

## DINO-WM: Zero-Shot World Models

### Leveraging Pre-trained Vision

DINO-WM (2024) demonstrates that **pre-trained visual representations** can enable world models without task-specific training.

### Core Innovation

Rather than learning representations from scratch, DINO-WM:

1. Uses frozen **DINOv2** embeddings as the observation space
2. Learns dynamics in this pre-trained feature space
3. Plans by optimizing actions to reach goal features

### Architecture

**Feature Extraction**:
$$
z_t = \text{DINOv2}(o_t) \quad \text{(frozen)}
$$

**Latent Dynamics Model**:
$$
\hat{z}_{t+1} = f_\theta(z_t, a_t)
$$

Implemented as a Vision Transformer operating on patch embeddings.

**Planning via MPC**:
$$
a^*_{0:H} = \arg\min_{a_{0:H}} \| \hat{z}_H - z_{\text{goal}} \|
$$

### Key Advantages

1. **Zero-shot Generalization**: Works on unseen tasks without retraining
2. **Offline Learning**: Trains from pre-collected trajectories
3. **No Reward Engineering**: Uses visual goal-reaching objective
4. **Scalable**: Benefits from larger pre-trained models

### Results

- 45% higher success rates than baselines
- 56% improvement in visual fidelity (LPIPS)
- Generalizes across mazes, manipulation, multi-object scenes

---

## Genie: Generative Interactive Environments

### World Models as Simulators

Google DeepMind's **Genie** (2024) takes world models to their logical conclusion: generating entire interactive environments from a single image.

### Architecture

Genie uses a three-component architecture:

**1. Video Tokenizer (ST-ViViT)**:
- Spatiotemporal Vision Transformer
- Converts video frames to discrete tokens
- Enables efficient sequence modeling

**2. Latent Action Model**:
- Infers actions from video without labels
- Learns a discrete action space automatically
- Enables controllable generation

**3. Dynamics Model**:
- Autoregressive transformer (like GPT)
- Predicts next frame tokens given history and action
- Generates consistent, controllable futures

### Training

Genie trains on **unlabeled internet videos**:
- No action labels required
- Learns controllable latent actions
- Scales to diverse domains

### Capabilities

- Generate playable environments from single images
- Create game-like interactions from photos or sketches
- Enable human or AI control of generated worlds
- Potential for training embodied agents

### Significance

Genie represents a path toward:
- **Foundation world models**: Pre-trained on diverse video
- **Imagination-based training**: Agents learn in generated worlds
- **Creative tools**: Interactive content from static images

---

## Cosmos: Physical World Simulation

### NVIDIA's World Foundation Models

**Cosmos** (2024) aims to build world models that understand physical reality:

### Key Components

**Tokenization**:
- Continuous tokens for high-fidelity reconstruction
- Discrete tokens for efficient scaling
- Joint image-video tokenizers

**World Model Architectures**:
- Autoregressive transformers
- Diffusion-based models
- Hybrid approaches

### Training Pipeline

1. **Video Curation**: High-quality, physically consistent videos
2. **Multi-stage Training**: Progressive resolution increase
3. **Physics Grounding**: Ensure physical plausibility

### Applications

- Autonomous vehicle simulation
- Robotic planning and training
- Synthetic data generation
- Game and content creation

---

## Theoretical Foundations: General Agents Contain World Models

A landmark paper by Richens, Abel, Bellot, and Everitt (ICML 2025) provides the first formal proof that **world models are not optional but necessary** for generally capable AI agents. This section explores this foundational result in depth.

### The Central Question

A long-standing debate in AI research concerns whether agents need internal models of their environment:

- **Model-based view**: Agents must learn predictive models to plan and generalize
- **Model-free view**: Agents can learn effective behaviors through trial and error without explicit world models

The paper settles this debate definitively: **general agents necessarily contain world models**.

### Formal Framework

#### Definitions

An agent is defined as a policy $\pi: \mathcal{H} \rightarrow \Delta(\mathcal{A})$ mapping histories to action distributions.

A **world model** is a learned representation that predicts:
$$
P(s_{t+1}, r_t | s_t, a_t)
$$
the next state and reward given current state and action.

An agent **generalizes** if it can achieve goals in novel situations not seen during training.

#### The Main Theorem

> **Theorem (Richens et al., 2025)**: Any agent capable of optimal behavior across a sufficiently diverse set of multi-step goal-directed tasks must have learned representations that constitute a world model of its environment.

More precisely, if an agent can:
1. Achieve arbitrary goals specified at test time
2. Generalize to new goal configurations
3. Plan over multiple time steps

Then the agent's internal representations **necessarily encode** predictive information about environment dynamics.

### Proof Sketch

The proof proceeds through several key steps:

**Step 1: Goal-Directed Behavior Requires Prediction**

To achieve a goal $g$ from state $s$, an agent must evaluate which actions lead toward $g$. This evaluation implicitly requires predicting the consequences of actions:

$$
\pi^*(a|s, g) \propto \sum_{s'} P(s'|s,a) \cdot V(s', g)
$$

**Step 2: Generalization Implies Structure**

If an agent generalizes to new goals, it cannot simply memorize state-action mappings. It must have learned structured representations that capture how actions affect states.

**Step 3: Structure Encodes Dynamics**

These structured representations, when analyzed formally, contain sufficient information to reconstruct environment dynamics—they **are** world models.

### Model Extraction

A remarkable corollary is that world models can be **extracted** from trained agents:

#### Extraction Algorithm

Given a trained policy $\pi$:

1. **Probe internal representations** at different layers
2. **Train a decoder** to predict next states from representations
3. **Verify predictions** match actual environment dynamics

The paper shows this extraction is always possible for general agents, and the extracted model's accuracy correlates with agent performance.

#### Implications for Interpretability

This provides a principled approach to understanding what agents have learned:

```
Trained Agent → Extract World Model → Analyze Predictions
                                           ↓
                              Understand Agent Beliefs
```

### Scaling Laws for World Models

The paper establishes a fundamental relationship:

> **Corollary**: The complexity of achievable goals is bounded by world model accuracy.

$$
\text{Goal Complexity} \leq f(\text{World Model Accuracy})
$$

This means:
- Simple goals (single-step) require minimal world modeling
- Complex goals (multi-step, compositional) require accurate world models
- **To build more capable agents, we must build better world models**

### Implications for AI Development

#### 1. Architecture Design

The theorem suggests architectures should explicitly support world modeling:

| Approach | Alignment with Theory |
|:---------|:---------------------|
| Model-based RL | Direct implementation |
| Transformers (next-token prediction) | Implicit world modeling |
| Retrieval-augmented systems | External world model |
| Pure model-free RL | Will converge to implicit world model |

#### 2. Safety and Alignment

Understanding agent world models enables:

- **Detecting misalignment**: Check if world model matches reality
- **Identifying failure modes**: Find where predictions break down
- **Improving robustness**: Correct world model errors before deployment

#### 3. Capability Evaluation

World model quality provides a **proxy for capability**:

$$
\text{Agent Capability} \approx g(\text{World Model Quality})
$$

This enables evaluating agents without exhaustive task testing.

#### 4. Training Objectives

The theorem suggests world model learning should be an explicit objective:

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{world\_model}
$$

Many successful approaches (Dreamer, JEPA) already do this.

### Connection to Large Language Models

The paper has profound implications for LLMs:

#### LLMs as World Models

Next-token prediction implicitly learns world models:

$$
P(x_{t+1} | x_{1:t}) \approx \text{World Model Prediction}
$$

When trained on text describing the world, LLMs learn to predict how the world works.

#### Emergent Planning

This explains emergent capabilities in LLMs:
- **Chain-of-thought reasoning**: Explicit simulation using world model
- **In-context learning**: Rapid world model adaptation
- **Tool use**: Extending world model with external dynamics

#### Limitations Explained

It also explains LLM failures:
- **Hallucinations**: World model makes incorrect predictions
- **Planning failures**: World model lacks relevant dynamics
- **Physical reasoning**: Insufficient training on physical dynamics

### The Model-Free Paradox Resolved

How do apparently "model-free" algorithms like DQN succeed?

The paper resolves this paradox:

> Even model-free algorithms, when successful at general tasks, have **implicitly learned world models** in their value function representations.

The Q-function encodes predictive information:
$$
Q(s, a) = \mathbb{E}\left[\sum_t \gamma^t r_t | s_0=s, a_0=a\right]
$$

This expectation over future rewards requires (implicit) knowledge of dynamics.

### Future Research Directions

The paper opens several research directions:

1. **World Model Metrics**: Develop better measures of world model quality
2. **Extraction Methods**: Improve techniques for extracting world models
3. **Architecture Search**: Design architectures that learn better world models
4. **Transfer Learning**: Use extracted world models for domain transfer
5. **Safety Applications**: Apply world model analysis to AI safety

### Summary

The "General Agents Contain World Models" paper provides:

| Contribution | Significance |
|:-------------|:-------------|
| **Necessity proof** | World models required for general intelligence |
| **Extraction method** | Can recover world models from any agent |
| **Scaling law** | Goal complexity bounded by model accuracy |
| **Unification** | Bridges model-based and model-free approaches |

This theoretical foundation validates decades of world model research and provides clear guidance for building more capable AI systems.

---

## Comparison of World Model Approaches

| Approach | Representation | Prediction Target | Training Data | Key Strength |
|:---------|:---------------|:------------------|:--------------|:-------------|
| **Ha & Schmidhuber** | VAE latents | Next latent state | Task-specific | Simplicity, dream training |
| **Dreamer** | RSSM states | Latent + reward | Task-specific | Sample efficiency |
| **I-JEPA** | ViT features | Masked patches | Images | Semantic representations |
| **V-JEPA** | Temporal features | Future frames | Videos | Temporal understanding |
| **DINO-WM** | Frozen DINOv2 | Next features | Offline data | Zero-shot generalization |
| **Genie** | Discrete tokens | Next frame tokens | Unlabeled video | Generative worlds |
| **Cosmos** | Hybrid tokens | Video continuation | Curated video | Physical realism |

---

## Progressive Refinement: From Coarse to Fine

Modern world models increasingly adopt **hierarchical** or **progressive** prediction:

### Multi-Scale Representations

Different levels capture different abstractions:
- **Low-level**: Pixel dynamics, textures
- **Mid-level**: Object motion, interactions
- **High-level**: Scene semantics, goals

### Coarse-to-Fine Generation

1. Predict high-level plan/trajectory
2. Refine with mid-level details
3. Fill in low-level specifics

This mirrors how humans plan: abstract goals → concrete actions → motor execution.

### Benefits

- **Computational efficiency**: Plan at appropriate resolution
- **Long-horizon reasoning**: Abstract away irrelevant details
- **Robust planning**: High-level plans transfer better

---

## Future Directions

### Foundation World Models

Pre-trained world models that:
- Learn from internet-scale video
- Transfer to diverse downstream tasks
- Enable few-shot adaptation

### Physical Understanding

World models that:
- Respect conservation laws
- Model rigid-body dynamics
- Handle deformable objects
- Understand causality

### Multi-Modal Worlds

Integrating:
- Vision, audio, touch
- Language grounding
- Social dynamics

### Embodied Training

Using world models for:
- Sim-to-real transfer
- Safe exploration
- Skill learning

---

## Summary

World models have evolved from a compelling idea to a practical paradigm:

| Era | Key Development |
|:----|:----------------|
| **2018** | Ha & Schmidhuber: Learning to dream |
| **2019-2023** | Dreamer series: Scalable imagination |
| **2022-2023** | JEPA: Representation prediction |
| **2024** | DINO-WM: Zero-shot with frozen features |
| **2024** | Genie: Generative interactive worlds |
| **2024** | Cosmos: Physical world simulation |
| **2025** | Theoretical proof of necessity |

The trajectory is clear: world models are not just useful—they are essential for building generally capable AI systems.

---

*For further reading: Ha & Schmidhuber "World Models" (2018), LeCun "A Path Towards Autonomous Machine Intelligence" (2022), Richens et al. "General Agents Contain World Models" (2025).*
